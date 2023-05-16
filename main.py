from escnn.group import *

G = cyclic_group(N=8)

print(G.order())

print(G.identity)

import os
import time

from src import EXPERIMENT_PATH
from src.model.resnet import ResNet
from src.data_module import FlightTrajectoryDataModule

test_scenario = FlightTrajectoryDataModule().test_dataloader().dataset[0]

# iteratre through all experiments in the experiment folder
for experiment in [os.path.join(EXPERIMENT_PATH, experiment) for experiment in os.listdir(EXPERIMENT_PATH)]:
    # load each version of this experiment
    for version in os.listdir(experiment):
        # get path to the latest checkpoint
        checkpoint_dir = os.path.join(experiment, version, 'checkpoints')
        # load the model weights from the latest checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, os.listdir(checkpoint_dir)[0])
        # load the model weights
        print('Loading model from checkpoint: {}'.format(checkpoint_path))
        model = ResNet.load_from_checkpoint(checkpoint_path)
        prediction = model.predict(test_scenario)
        


        # plot the trajectory of the model
        plot_scenario_topdown(test_scenario, prediction)
        time.sleep(3)