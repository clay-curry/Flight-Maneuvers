# path pointing to a dataset of experiment logs, consisting of checkpoints and tensorboard logs
EXPERIMENT_DIR = 'experiments'
# path pointing to a dataset of simulated flight trajectories with labeled maneuvers
TRAIN_DATA_DIR = 'maneuver_dataset'
# categorical maneuvers assigned to the projectile at each timestep
MANEUVERS = ['takeoff', 'turn', 'line', 'orbit', 'landing']
# default features
FEATURES = ['pos', 'vel', 'alt', 'dpos', 'dvel', 'maneuver']
# number of workers for dataloading
NUM_DATALOADERS = 8
# sampling rate
SAMPLING_PERIOD = 60
# maximum length of a trajectory
MAX_TIMESTEPS = 256

# logger
LOGGER = TBLogger
# use Automatic Mixed Precision
AMP = False
# compute loss with Cross Entropy
LOSS = torch.nn.CrossEntropyLoss()
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
CKPT_INTERVAL = -1
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
