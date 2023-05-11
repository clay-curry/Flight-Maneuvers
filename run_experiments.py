from anne.model.resnet import ResNet
from anne.dataset import FlightTrajectoryDataModule

from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import LearningRateMonitor

# set of simulated flight trajectories with labeled maneuvers
DATASET_PATH = 'examples/maneuver_dataset'

# provides a location for storing and loading experiment results
LOGGING_PATH = 'experiments'

# trajectories are observed once per SAMPLING_INTERVAL seconds
SAMPLING_INTERVAL = 30  

# number of validation and test examples to use per epoch
NUM_VALID = 5
NUM_TEST = 1
MAX_STEPS = 50 * 100


for NUM_TRAIN in [1, 5, 10, 20, 50, 100]:

    dataset = FlightTrajectoryDataModule(
        DATASET_PATH,
        num_train=NUM_TRAIN, 
        num_valid=NUM_VALID, 
        num_test=NUM_TEST, 
        batch_size=1
    )

    model = ResNet(
        k_size=[3] * 300, 
        c_hidden=[36] * 300,
        d_size=[5] * 300,
        block_type="PreActResNetBlock"
    )

    logger = TensorBoardLogger(LOGGING_PATH, name="resnet" + '-train-' + str(NUM_TRAIN))
    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = Trainer(logger=logger, callbacks=[lr_monitor], log_every_n_steps=10, max_steps=MAX_STEPS)
    trainer.fit(model, dataset)


    """
    model = SE2_ResNet(
        k_size=[3] * 300, 
        c_hidden=[36] * 300,
        d_size=[5] * 300,
        block_type="PreActResNetBlock"
    )

    logger = TensorBoardLogger(LOGGING_PATH, name="se2_resnet" + '-train-' + NUM_TRAIN)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = Trainer(logger=logger, callbacks=[lr_monitor], log_every_n_steps=10, max_epochs=NUM_EPOCHS)
    trainer.fit(model, dataset)


    model = SE3_ResNet(
        k_size=[3] * 300, 
        c_hidden=[36] * 300,
        d_size=[5] * 300,
        block_type="PreActResNetBlock"
    )

    logger = TensorBoardLogger(LOGGING_PATH, name="se3_resnet" + '-train-' + NUM_TRAIN)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = Trainer(logger=logger, callbacks=[lr_monitor], log_every_n_steps=10, max_epochs=NUM_EPOCHS)
    trainer.fit(model, dataset)
    """