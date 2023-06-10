# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

"""Input/output checkpointing."""

import os
import random
import sys
import numpy as np

import torch


def ensure_directory_exists(filename):
    """Build filename's path if it does not already exists."""
    dirname = os.path.dirname(filename)
    os.makedirs(dirname, exist_ok = True)

def save_checkpoint(iteration, model, optimizer, opt_param_scheduler):
    """Save a model checkpoint."""

    print('saving checkpoint at iteration {:7d} to {}'.format(iteration, args.save))


    # Checkpoint name.
    checkpoint_name = get_checkpoint_name(args.save, iteration)

    # Save distributed optimizer's custom parameter state.
    if args.use_distributed_optimizer:
        optim_checkpoint_name = \
            get_distributed_optimizer_checkpoint_name(checkpoint_name)
        ensure_directory_exists(optim_checkpoint_name)
        optimizer.save_parameter_state(optim_checkpoint_name)

    # Collect args, model, RNG.
    if not torch.distributed.is_initialized() \
       or mpu.get_data_parallel_rank() == 0:

        # Arguments, iteration, and model.
        state_dict = {}
        state_dict['args'] = args
        state_dict['checkpoint_version'] = 3.0
        state_dict['iteration'] = iteration
        if len(model) == 1:
            state_dict['model'] = model[0].state_dict_for_save_checkpoint()
        else:
            for i in range(len(model)):
                mpu.set_virtual_pipeline_model_parallel_rank(i)
                state_dict['model%d' % i] = \
                    model[i].state_dict_for_save_checkpoint()

        # Optimizer stuff.
        if not args.no_save_optim:
            if optimizer is not None:
                state_dict['optimizer'] = optimizer.state_dict()
            if opt_param_scheduler is not None:
                state_dict['opt_param_scheduler'] = \
                    opt_param_scheduler.state_dict()

        # RNG states.
        if not args.no_save_rng:
            state_dict["rng_state"] = rng_state

        # Save.
        ensure_directory_exists(checkpoint_name)
        torch.save(state_dict, checkpoint_name)

    # Wait so everyone is done (necessary)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    print_rank_0('  successfully saved checkpoint at iteration {:7d} to {}' \
                 .format(iteration, args.save))

    # And update the latest iteration
    if not torch.distributed.is_initialized() \
       or torch.distributed.get_rank() == 0:
        tracker_filename = get_checkpoint_tracker_filename(args.save)
        with open(tracker_filename, 'w') as f:
            f.write(str(iteration))

    # Wait so everyone is done (not necessary)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


def _load_base_checkpoint(load_dir, rank0=False):
    """ Load the base state_dict from the given directory

    If rank0 is true, just loads rank 0 checkpoint, ignoring arguments.
    """

    # Read the tracker file and set the iteration.
    tracker_filename = get_checkpoint_tracker_filename(load_dir)

    # If no tracker file, return nothing
    if not os.path.isfile(tracker_filename):
        if not rank0:
            print_rank_0('WARNING: could not find the metadata file {} '.format(
                tracker_filename))
            print_rank_0('    will not load any checkpoints and will start from '
                         'random')
        return None, False

    # Otherwise, read the tracker file and either set the iteration or
    # mark it as a release checkpoint.
    iteration, release = read_metadata(tracker_filename)

    # Checkpoint.
    if rank0:
        checkpoint_name = find_checkpoint_rank_0(load_dir, iteration, release)
    else:
        checkpoint_name = get_checkpoint_name(load_dir, iteration, release)
        if release:
            print_rank_0(f' loading release checkpoint from {load_dir}')
        else:
            print_rank_0(f' loading checkpoint from {load_dir} at iteration {iteration}')

    # Load the checkpoint.
    try:
        state_dict = torch.load(checkpoint_name, map_location='cpu')
    except ModuleNotFoundError:
        from megatron.fp16_deprecated import loss_scaler
        # For backward compatibility.
        if not rank0:
            print_rank_0(' > deserializing using the old code structure ...')
        sys.modules['fp16.loss_scaler'] = sys.modules[
            'megatron.fp16_deprecated.loss_scaler']
        sys.modules['megatron.fp16.loss_scaler'] = sys.modules[
            'megatron.fp16_deprecated.loss_scaler']
        state_dict = torch.load(checkpoint_name, map_location='cpu')
        sys.modules.pop('fp16.loss_scaler', None)
        sys.modules.pop('megatron.fp16.loss_scaler', None)
    except BaseException as e:
        print_rank_0('could not load the checkpoint')
        print_rank_0(e)
        sys.exit()

    return state_dict, release


def load_args_from_checkpoint(args, load_arg='load'):
    """Set required arguments from the checkpoint specified in the
    arguments.

    Will overwrite arguments that have a non-None default value, but
    will leave any arguments that default to None as set.

    Returns the same args NameSpace with the new values added/updated.

    If no checkpoint is specified in args, or if the checkpoint is
    there but invalid, the arguments will not be modified

    """
    load_dir = getattr(args, load_arg)

    if load_dir is None:
        print('No load directory specified, using provided arguments.')
        return args

    state_dict, release = _load_base_checkpoint(load_dir, rank0=True)

    # Args.
    if not state_dict:
        print('Checkpoint not found to provide arguments, using provided arguments.')
        return args

    if 'args' not in state_dict:
        print('Checkpoint provided does not have arguments saved, using provided arguments.')
        return args

    checkpoint_args = state_dict['args']
    checkpoint_version = state_dict.get('checkpoint_version', 0)
    args.iteration = state_dict['iteration']



def load_checkpoint(model, optimizer, opt_param_scheduler, load_arg='load', strict=True):
    """Load a model checkpoint and return the iteration.
    strict (bool): whether to strictly enforce that the keys in
        :attr:`state_dict` of the checkpoint match the names of
        parameters and buffers in model.
    """
    load_dir = None

    model = unwrap_model(model)

    state_dict, release = _load_base_checkpoint(load_dir, rank0=False)

    # Checkpoint not loaded.
    if state_dict is None:

        # Conditionally exit at this point.
        if args.exit_on_missing_checkpoint:
            print_rank_0(">> '--exit-on-missing-checkpoint' set ... exiting. <<")
            torch.distributed.barrier()
            sys.exit()

        # Iteration defaults to 0.
        return 0

    # Set checkpoint version.
    set_checkpoint_version(state_dict.get('checkpoint_version', 0))

    # Set iteration.
    if args.finetune or release:
        iteration = 0
    else:
        try:
            iteration = state_dict['iteration']
        except KeyError:
            try:  # Backward compatible with older checkpoints
                iteration = state_dict['total_iters']
            except KeyError:
                print_rank_0('A metadata file exists but unable to load '
                             'iteration from checkpoint {}, exiting'.format(
                                 checkpoint_name))
                sys.exit()

    # Check arguments.
    assert args.consumed_train_samples == 0
    assert args.consumed_valid_samples == 0
    if 'args' in state_dict and not args.finetune:
        checkpoint_args = state_dict['args']
        check_checkpoint_args(checkpoint_args)
        args.consumed_train_samples = getattr(checkpoint_args,
                                              'consumed_train_samples', 0)
        update_num_microbatches(consumed_samples=args.consumed_train_samples)
        args.consumed_valid_samples = getattr(checkpoint_args,
                                              'consumed_valid_samples', 0)
    else:
        print_rank_0('could not find arguments in the checkpoint ...')

    # Model.
    if len(model) == 1:
        model[0].load_state_dict(state_dict['model'], strict=strict)
    else:
        for i in range(len(model)):
            mpu.set_virtual_pipeline_model_parallel_rank(i)
            model[i].load_state_dict(state_dict['model%d' % i], strict=strict)

    # Fix up query/key/value matrix ordering if needed.
    checkpoint_version = get_checkpoint_version()
    print(f' checkpoint version {checkpoint_version}')
    fix_query_key_value_ordering(model, checkpoint_version)

    # Optimizer.
    if not release and not args.finetune and not args.no_load_optim:
        try:
            # Load state dict.
            if optimizer is not None:
                optimizer.load_state_dict(state_dict['optimizer'])

            # Load distributed optimizer's custom parameter state.
            if args.use_distributed_optimizer:
                tracker_filename = get_checkpoint_tracker_filename(load_dir)
                iteration, release = read_metadata(tracker_filename)
                model_checkpoint_name = \
                    get_checkpoint_name(load_dir, iteration, release)
                optim_checkpoint_name = \
                    get_distributed_optimizer_checkpoint_name(
                        model_checkpoint_name)
                optimizer.load_parameter_state(optim_checkpoint_name)

            # Load scheduler.
            if opt_param_scheduler is not None:
                if 'lr_scheduler' in state_dict: # backward compatbility
                    opt_param_scheduler.load_state_dict(state_dict['lr_scheduler'])
                else:
                    opt_param_scheduler.load_state_dict(state_dict['opt_param_scheduler'])
        except KeyError:
            print('Unable to load optimizer from checkpoint {}. '
                         'Specify --no-load-optim or --finetune to prevent '
                         'attempting to load the optimizer state, '
                         'exiting ...'.format(checkpoint_name))
            sys.exit()
    else:
        if (args.fp16 or args.bf16) and optimizer is not None:
            optimizer.reload_model_params()

    # rng states.
    if not release and not args.finetune and not args.no_load_rng:
        try:
            if 'rng_state' in state_dict:
                # access rng_state for data parallel rank
                if args.data_parallel_random_init:

                    rng_state = state_dict['rng_state'][mpu.get_data_parallel_rank()]
                else:
                    rng_state = state_dict['rng_state'][0]
                random.setstate(rng_state['random_rng_state'])
                np.random.set_state(rng_state['np_rng_state'])
                torch.set_rng_state(rng_state['torch_rng_state'])
                torch.cuda.set_rng_state(rng_state['cuda_rng_state'])
                # Check for empty states array
                if not rng_state['rng_tracker_states']:
                    raise KeyError
                tensor_parallel.get_cuda_rng_tracker().set_states(
                    rng_state['rng_tracker_states'])
            else:  # backward compatability
                random.setstate(state_dict['random_rng_state'])
                np.random.set_state(state_dict['np_rng_state'])
                torch.set_rng_state(state_dict['torch_rng_state'])
                torch.cuda.set_rng_state(state_dict['cuda_rng_state'])
                # Check for empty states array
                if not state_dict['rng_tracker_states']:
                    raise KeyError
                tensor_parallel.get_cuda_rng_tracker().set_states(
                    state_dict['rng_tracker_states'])
        except KeyError:
            print('Unable to load rng state from checkpoint {}. '
                         'Specify --no-load-rng or --finetune to prevent '
                         'attempting to load the rng state, '
                         'exiting ...'.format(checkpoint_name))
            sys.exit()

    # Some utilities want to load a checkpoint without distributed being initialized
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    print(f'  successfully loaded checkpoint from {args.load} '
                 f'at iteration {iteration}')

    return iteration

