import os
import torch
import random
import numpy as np
from itertools import tee
from functools import partial
from collections.abc import Mapping
from typing import Any, cast, Iterable, List, Literal, Optional, Tuple, Union

from flight_maneuvers.core.log import * 
from flight_maneuvers.core.checkpoint import load_checkpoint
from flight_maneuvers.core import default_hparams

def seed_everything(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class Trainer:
    def __init__(self, model_type, **hparams):
        # Update default hyperparameters
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Loads the most recent checkpoint from indicated version of `model_type`, or creates
        # a new model if no version is specified
        self.model, self.optimizer, self.data_module = load_checkpoint(model_type, **hparams)
        # Create logger
        self.log = hparams['LOGGER'](**hparams)
        self.loss = hparams['LOSS'].to(self.device)
    
    def add_hyperparameters():
        pass

    @classmethod
    def load_state_dict(cls):
        T = cls()
        def __init__(self, model_type, **hparams):
            pass


def sanity_check(self, model, optimizer, data_module):
    pass


def fit(self, model_type, stopping_condition, **hparams):
        # Update default hyperparameters
        # Loads the most recent checkpoint from indicated version of `model_type`, or creates 
        # a new model if no version is specified 
        self.model, self.optimizer, self.data_module = load_checkpoint(model_type, **hparams)        
        # Create logger
        self.log = hparams['LOGGER'](**hparams)

        train_loader = self.data_module.train_dataloader()
        val_loader = self.data_module.val_dataloader()

        # Train the model
        for epoch in range(self.hparams['MAX_EPOCHS']):
            # Train for one epoch
            self.train_epoch(self.model, train_loader, logger=self.log)
            # Validate the model
            self.validate_epoch(self.model, val_loader, logger=self.log)
            # Save the model
            self.save_checkpoint(self.model, self.optimizer, epoch, self.log)
            # Check if the stopping condition is met
            if stopping_condition(self.log):
                break

def train_epoch(model, train_loader, logger):
    model.train()
    loss, accuracy = 0, 0
    for idx, batch in enumerate(train_loader):
        # Train the model
        train_step(model, batch)
        # Log metrics
        logger.add_scalar('Loss/train', loss, idx)
        logger.add_scalar('Accuracy/train', accuracy, idx)
     

def train_step(self, batch):
        inputs, targets = batch
        self.optimizer.zero_grad()
        logits = self.model(inputs)
        loss = self.hparams['LOSS'](logits, targets)
        loss.backward()        
        self.optimizer.step()
        accuracy = (logits.argmax(dim=-1) == targets).float().mean()        
        return loss, accuracy

def validate_step(self, batch):
        inputs, targets = batch
        logits = self.model(inputs)
        loss = self.hparams['LOSS'](logits, targets)
        accuracy = (logits.argmax(dim=-1) == targets).float().mean()
        return {'Loss/val': loss, 'Accuracy/val': accuracy}

# If the selected scheduler is a ReduceLROnPlateau scheduler.
