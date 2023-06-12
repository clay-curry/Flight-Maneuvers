import argparse
import pathlib

def get_parser():
    PARSER = argparse.ArgumentParser(description='Flight Maneuvers')

    paths = PARSER.add_argument_group('Paths')
    paths.add_argument('--data_dir', type=pathlib.Path, default=pathlib.Path('./data'),
                    help='Directory where the data is located or should be downloaded')
    paths.add_argument('--log_dir', type=pathlib.Path, default=pathlib.Path('./results'),
                    help='Directory where the results logs should be saved')
    paths.add_argument('--dllogger_name', type=str, default='dllogger_results.json',
                    help='Name for the resulting DLLogger JSON file')
    paths.add_argument('--save_ckpt_path', type=pathlib.Path, default=None,
                    help='File where the checkpoint should be saved')
    paths.add_argument('--load_ckpt_path', type=pathlib.Path, default=None,
                    help='File of the checkpoint to be loaded')

    optimizer = PARSER.add_argument_group('Optimizer')
    optimizer.add_argument('--optimizer', choices=['adam', 'sgd', 'lamb'], default='adam')
    optimizer.add_argument('--learning_rate', '--lr', dest='learning_rate', type=float, default=0.002)
    optimizer.add_argument('--min_learning_rate', '--min_lr', dest='min_learning_rate', type=float, default=None)
    optimizer.add_argument('--momentum', type=float, default=0.9)
    optimizer.add_argument('--weight_decay', type=float, default=0.1)

    PARSER.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    PARSER.add_argument('--batch_size', type=int, default=240, help='Batch size')
    PARSER.add_argument('--seed', type=int, default=None, help='Set a seed globally')
    PARSER.add_argument('--num_workers', type=int, default=8, help='Number of dataloading workers')

    PARSER.add_argument('--amp', type=str2bool, nargs='?', const=True, default=False, help='Use Automatic Mixed Precision')
    PARSER.add_argument('--gradient_clip', type=float, default=None, help='Clipping of the gradient norms')
    PARSER.add_argument('--accumulate_grad_batches', type=int, default=1, help='Gradient accumulation')
    PARSER.add_argument('--ckpt_interval', type=int, default=-1, help='Save a checkpoint every N epochs')
    PARSER.add_argument('--eval_interval', dest='eval_interval', type=int, default=20,
                        help='Do an evaluation round every N epochs')
    PARSER.add_argument('--silent', type=str2bool, nargs='?', const=True, default=False,
                        help='Minimize stdout output')
    PARSER.add_argument('--wandb', type=str2bool, nargs='?', const=True, default=False,
                        help='Enable W&B logging')

    PARSER.add_argument('--benchmark', type=str2bool, nargs='?', const=True, default=False,
                        help='Benchmark mode')

    return PARSER