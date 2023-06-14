# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

"""Input/output checkpointing."""

import os
import json
import torch
import torch

def load_checkpoint(**hparams):
    """Load a model checkpoint and return the iteration.
    strict (bool): whether to strictly enforce that the keys in
        :attr:`state_dict` of the checkpoint match the names of
        parameters and buffers in model.
    """
    version = get_version_logs(model_type, hparams)
    
    model_type = hparams['MODEL']
    optimizer_type = hparams['OPTIMIZER']
    scheduler_type = hparams['SCHEDULER']

    version_path = os.path.join('runs', model_type.__qualname__, ' - ', version)
    ckpt_path = os.path.join(version_path, 'checkpoint.pt')
    if os.path.exists(ckpt_path):
        # if a checkpoint exists, load the model, optimizer, and datamodule
        model_state = torch.load(os.path.join(ckpt_path, 'model.pt'))
        optimizer_state = torch.load(os.path.join(ckpt_path, 'optimizer.pt'))
        model = model_type.load_state_dict(model_state),
        optimizer = optimizer_type.load_state_dict(optimizer_state),
        return model, optimizer, scheduler

    else:
        # if no checkpoint exists, create a new model, optimizer, and datamodule
        model = model_type(**hparams)
        optimizer = optimizer_type(model.parameters(), **hparams)
        scheduler = scheduler_type(optimizer, **hparams)
        return model, optimizer, scheduler

def save_model_checkpoint(model, datamodule, optimizer, hparams):
    """Save a model checkpoint."""
    to_save = { 'model': model.state_dict(), 'datamodule': datamodule.state_dict(), 'optimizer': optimizer.state_dict() }
    version = get_hparam_logdir(model, hparams)
    torch.save(to_save, os.path.join(version, 'model.pt'))

def get_hparam_logdir(**hparams):
    # create a directory for the model type
    version_number = 0
    model_path = os.path.join('runs', hparams['MODEL'].__qualname__)
    os.makedirs(model_path, exist_ok=True)
    for path in os.listdir(model_path):
        # get version number from the rightmost string of digits
        version_number = int(path.split(' - ')[-1])
        if json.load(os.path.join(model_path, path)) == json.loads(hparams):
            return os.path.join(model_path, ' - ', path)
    return os.path.join(model_path, ' - ', str(version_number + 1))        
    
