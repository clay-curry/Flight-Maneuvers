import torch
import lightning as L
from itertools import chain, combinations, product

from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping

from flight_maneuvers.modules.resnet import ResNet
from flight_maneuvers.adapter import LigntingAdapter
from flight_maneuvers.data_module import FlightTrajectoryDataModule

def train(model: torch.nn.Module, hparams):
    LOG_DIR = 'logs'
    MAX_STEPS = 250
    VALIDATE_EVERY_N_STEPS = 50
    for num_train in [1, 5, 10, 20, 100]:
        L.seed_everything(0)
        data_module = FlightTrajectoryDataModule(num_train=num_train, num_valid=15, num_test=100)
        early_stopping = EarlyStopping(monitor='val_loss', mode='max', patience=15)
        litmodel = LigntingAdapter(model, **hparams)
        tb_logger = TensorBoardLogger(LOG_DIR, name='resnet-'+str(num_train)+'-train')
        trainer = L.Trainer(
            tb_logger, 
            callbacks=[early_stopping], 
            max_steps=MAX_STEPS, 
            check_val_every_n_epoch=None,
            val_check_interval=VALIDATE_EVERY_N_STEPS
        )
        trainer.fit(litmodel, data_module)
        trainer.test(litmodel, datamodule=data_module)
        


def powerset(iterable):
    "list(powerset([1,2,3])) --> [(), (1,), (2,), (3,), (1,2), (1,3), (2,3), (1,2,3)]"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


optimizer_space = {
    'optimizer': [ 
            {'opt': 'Adam', 'lr': 1e-3}, {'opt': 'Adam', 'lr': 1e-4}, {'opt': 'Adam', 'lr': 1e-5},
            {'opt': 'SGD' , 'lr': 1e-3}, {'opt': 'SGD' , 'lr': 1e-4}, {'opt': 'SGD' , 'lr': 1e-5},
        ],
    'lr_scheduler': [
            {'lrs': 'StepLR', 'step_size': 100 }, 
            {'lrs': 'StepLR', 'step_size': 200 },
            {'lrs': 'StepLR', 'step_size': 500 },
            {'lrs': 'ReduceLROnPlateau', 'min_lr': 1e-5}
        ],
}

model_spaces = {
    ResNet: {
        "features": [
                ['dpos'], ['vel', 'dpos'], 
                ['vel', 'dpos', 'dvel'],
                ['pos', 'vel', 'dpos', 'dvel']
            ],
        "c_hidden": [[16]*30, [16]*30 + [32]*30, [16]*30 + [32]*30 + [64]*30, [16]*30 + [32]*30 + [64]*30 + [128]*30],
        "kernel_size": [3, 5],
        "act_fn_name": ['relu', 'leakyrelu', 'tanh', 'gelu'],
        "block_name": ['PreActResNetBlock', 'ResNetBlock'],
    }
}

def iterate_dict(dictionary):
    keys = dictionary.keys()
    values = dictionary.values()

    # Use itertools.product to generate combinations of values
    return iter(dict(zip(keys, combination)) for combination in product(*values))


def search_space(model):
    hparams = { **model_spaces[model], **optimizer_space }
    for hp in iterate_dict(hparams):
        train(model, hp)
