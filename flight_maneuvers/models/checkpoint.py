# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

"""Input/output checkpointing."""

import os
import random
import sys
import numpy as np
import json
import torch
from flight_maneuvers.data.datamodule import FlightTrajectoryDataModule

import torch

def load_checkpoint(model_type, **hparams):
    """Load a model checkpoint and return the iteration.
    strict (bool): whether to strictly enforce that the keys in
        :attr:`state_dict` of the checkpoint match the names of
        parameters and buffers in model.
    """
    optimizer_type = hparams['OPTIMIZER']
    datamodule_type = hparams['DATAMODULE']
    version = get_version_logs(model_type, hparams)
    version_path = os.path.join(hparams['EXPERIMENT_DIR'], model_type.__qualname__, ' - ', version)
    ckpt_path = os.path.join(version_path, 'checkpoint.pt')
    if os.path.exists(ckpt_path):
        # if a checkpoint exists, load the model, optimizer, and datamodule
        model_state = torch.load(os.path.join(ckpt_path, 'model.pt'))
        optimizer_state = torch.load(os.path.join(ckpt_path, 'optimizer.pt'))
        datamodule_state = torch.load(os.path.join(ckpt_path, 'datamodule.pt'))
        model = model_type.load_state_dict(model_state),
        optimizer = optimizer_type.load_state_dict(optimizer_state),
        datamodule = datamodule_type.load_state_dict(datamodule_state)
        return model, optimizer, datamodule           

    else:
        # if no checkpoint exists, create a new model, optimizer, and datamodule
        model = model_type(**hparams),
        optimizer = optimizer_type(model.parameters(), **hparams),
        datamodule = datamodule_type(**hparams)
        return model, optimizer, datamodule

def save_checkpoint(model, datamodule, optimizer, hparams):
    """Save a model checkpoint."""
    version = get_version_logs(model, hparams)
    torch.save(model.state_dict(), os.path.join(version, 'model.pt'))
    torch.save(optimizer.state_dict(), os.path.join(version, 'optimizer.pt'))
    torch.save(datamodule.state_dict(), os.path.join(version, 'datamodule.pt'))
        
def get_version_logs(model, hparams):
    # combine model name and experiment dir to create an unambiguous save location
    model_path = get_model_logs(model, hparams)
    # use hyperparameters to check if learner already exists 
    existing_versions = sorted(os.listdir(model_path))
    for v, version in enumerate(existing_versions):
        if json.load(open(os.path.join(model_path, version, 'hparams.json'))) == hparams:
            return version
    # if learner does not exist, create a new version
    version = f'version_{len(existing_versions)}'
    os.makedirs(os.path.join(model_path, version), exist_ok=True)
    return version

def get_model_logs(model, hparams):
    # combine model name and experiment dir to create an unambiguous save location
    model_path = os.path.join(hparams['EXPERIMENT_DIR'], model.__qualname__)
    os.makedirs(model_path, exist_ok=True)
    return model_path
    
