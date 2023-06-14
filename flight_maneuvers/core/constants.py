import torch
from flight_maneuvers.models.gnn.resnet import ResNet
from flight_maneuvers.data.datamodule import FlightTrajectoryDataModule

# seed
SEED = 0
# model to use
MODEL=ResNet
# categorical maneuvers assigned to the projectile at each timestep,
DATAMODULE = FlightTrajectoryDataModule
# number of workers for dataloading
NUM_DATALOADERS = 8
# sampling rate
SAMPLING_PERIOD = 60
# maximum length of a trajectory
MAX_TIMESTEPS = 256
# use Automatic Mixed Precision
AMP = False
# compute loss with Cross Entropy
LOSS = torch.nn.functional.cross_entropy
# number of trajectories used to train the model
TRAIN_SIZE = 30
# number of trajectories used to validate the model
VAL_SIZE = 10
# number of trajectories used to test the model
TEST_SIZE = 50
# number of trajectories used to update gradients
BATCH_SIZE = 10
# maximum number of epochs to train the model
MAX_EPOCHS = 1000
# clip gradients to this value
GRAD_CLIP = 1.0
# wait this many batches before stepping the optimizer
GRAD_ACCUM = 1
# checkpoint every this many epochs
CKPT_INTERVAL = 1
# validate every this many epochs
VALID_INTERVAL = 1
# validate every this many epochs
TEST_INTERVAL = 1
# optimizer to use
OPTIMIZER = torch.optim.Adam
# learning rate scheduler to use
LR_SCHEDULER = torch.optim.lr_scheduler.ReduceLROnPlateau
# initial learning rate
LR = 1e-3
# momentum for SGD
MOMENTUM = 0.9
# weight decay for SGD
WEIGHT_DECAY = 1e-4

default_hparams = {
    'SEED': SEED,
	'DATAMODULE': DATAMODULE,
    'MODEL': MODEL,
    'NUM_DATALOADERS': NUM_DATALOADERS,
	'SAMPLING_PERIOD': SAMPLING_PERIOD,
	'MAX_TIMESTEPS': MAX_TIMESTEPS,
	'AMP': AMP,
	'LOSS': LOSS,
	'TRAIN_SIZE': TRAIN_SIZE,
	'VAL_SIZE': VAL_SIZE,
	'TEST_SIZE': TEST_SIZE,
	'BATCH_SIZE': BATCH_SIZE,
	'MAX_EPOCHS': MAX_EPOCHS,
	'GRAD_CLIP': GRAD_CLIP,
	'GRAD_ACCUM': GRAD_ACCUM,
	'CKPT_INTERVAL': CKPT_INTERVAL,
	'VALID_INTERVAL': VALID_INTERVAL,
	'TEST_INTERVAL': TEST_INTERVAL,
	'OPTIMIZER': OPTIMIZER,
	'LR_SCHEDULER': LR_SCHEDULER,
	'LR': LR,
	'MOMENTUM': MOMENTUM,
	'WEIGHT_DECAY': WEIGHT_DECAY
}
