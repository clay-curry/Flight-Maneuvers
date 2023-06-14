""" provides subroutines that optimizes model hyperparameters a target metric

hyperparameter optimization is a search problem, where the search space is the space of hyperparameters
to optimize performance. Especially

occurs by a series of trials for each choice of parameter
"""
import torch
import random
import numpy as np
from abc import ABC, abstractmethod

from flight_maneuvers.nn.train import Trainer
from flight_maneuvers.data.datamodule import FlightTrajectoryDataModule

NUM_SEEDS = 30

def seed_everything(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def tune(model, sampler, target_metric, number_seeds=NUM_SEEDS, max_trials=100):
    """ tune the parameters of the model


    """
    best_hparams = None
    best_metric = metric = 0
    trial_num = 0
    while metric < target_metric and trial_num < max_trials:
        trial += 1
        # sample a set of hyperparameters
        hparams = sampler.sample()
        # train the model with the set of hyperparameters
        metric = expected_test_error(model, hparams, trial)
        # update the best set of hyperparameters if the metric is better
        if best_metric < metric:
            best_hparams = hparams
            best_metric = metric

    return best_hparams


def expected_test_error(model, hparams, number_seeds, trail_number):
    """ calculate the expected test error of the model
    """
    for seed in range(number_seeds):
        test_error(model, hparams, seed, trail_number)

def test_error(model, hparams, nsteps, seed, trail_number):
    """ calculate the test error of the model for a model trained with a test set produced by seed

    Err_T = E[L(y, f(x)) | T]

    where L is the loss function, f is the model trained on fixed training set T,
    and y and x are drawn randomly from some distribution P(y, x)
    """
    # set the seed
    seed_everything(seed)
    datamodule = FlightTrajectoryDataModule(**hparams)
    model = model(**hparams)
    trainer = Trainer(model, datamodule, **hparams)
    trainer.fit()



class BaseModelSampler(ABC):
    @abstractmethod
    def sample(self):
        raise NotImplementedError
