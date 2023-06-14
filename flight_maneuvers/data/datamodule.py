import os
import torch
import random
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

from flight_maneuvers.data.preprocessing import *

MANEUVERS = ['takeoff', 'turn', 'line', 'orbit', 'landing']

def transfer_to_accelerator(segs):
    states = []
    maneuvers = []
    for seg in segs:
        states.append(torch.from_numpy(seg[0]).device)
        maneuvers.append(torch.from_numpy(seg[1]))
    
    #states = torch.nested.nested_tensor(states)
    #maneuvers = torch.nested.nested_tensor(maneuvers)
    return states, maneuvers

# each training example consists of a variable-length, simulated flight trajectory with labeled maneuvers at each timestep
class FlightTrajectoryDataset(Dataset):
    
    def __init__(self, files, sampling_period, max_timesteps) -> None:
        ntimesteps = {f: pd.read_csv(f)['t'].shape[0] for f in files}
        self.files = sorted(files, key=lambda x: ntimesteps[x])
        self.sampling_period = sampling_period
        self.max_timesteps = max_timesteps
        
    def __len__(self):  return len(self.files)

    def __getitem__(self, scenario_idx):  
        instance = sample_timeseries(
            self.files[scenario_idx], 
            self.sampling_period,
            self.max_timesteps,
        )
        states, maneuvers = get_states_maneuvers(instance)
        states = states.to_numpy(dtype='float32')
        maneuvers = maneuvers.map(MANEUVERS.index).to_numpy(dtype='int64')
        return states, maneuvers

class FlightTrajectoryDataModule:
    
    def __init__(self, **kwargs) -> None:
        kwargs.setdefault('TRAIN_DATA_DIR', 'maneuver_dataset')
        num_valid = kwargs['VAL_SIZE']
        num_test = kwargs['TEST_SIZE']
        num_train = kwargs['TRAIN_SIZE']
        self.seed = kwargs['SEED']
        self.batch_size = kwargs['BATCH_SIZE']
        self.sampling_period = kwargs['SAMPLING_PERIOD']
        self.num_dataloaders = kwargs['NUM_DATALOADERS']
        self.max_timesteps = kwargs['MAX_TIMESTEPS']
        files = os.listdir(kwargs['TRAIN_DATA_DIR'])
        assert len(files) >= num_train + num_valid + num_test
        splits = np.cumsum([num_train, num_valid, num_test])
        files = [f for f in np.split(random.sample(files, len(files)), splits)][:-1]
        self.train_set = FlightTrajectoryDataset([os.path.join(kwargs['TRAIN_DATA_DIR'], f) for f in files[0]], kwargs['SAMPLING_PERIOD'], kwargs['MAX_TIMESTEPS'])
        self.val_set = FlightTrajectoryDataset([os.path.join(kwargs['TRAIN_DATA_DIR'], f) for f in files[1]], kwargs['SAMPLING_PERIOD'], kwargs['MAX_TIMESTEPS'])
        self.test_set = FlightTrajectoryDataset([os.path.join(kwargs['TRAIN_DATA_DIR'], f) for f in files[2]], kwargs['SAMPLING_PERIOD'], kwargs['MAX_TIMESTEPS'])

    def train_dataloader(self):
        return DataLoader(
            self.train_set, 
            self.batch_size, 
            num_workers=self.num_dataloaders, 
            collate_fn=transfer_to_accelerator, 
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set, 
            self.batch_size, 
            num_workers=self.num_dataloaders, 
            collate_fn=transfer_to_accelerator,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set, 
            self.batch_size, 
            num_workers=self.num_dataloaders, 
            collate_fn=transfer_to_accelerator
        )
    
    def state_dict(self):
        return {
            'SEED': self.seed,
            'train_set': self.train_set.files,
            'val_set': self.val_set.files,
            'test_set': self.test_set.files,
            'batch_size': self.batch_size,
            'sampling_period': self.sampling_period,
            'num_dataloaders': self.num_dataloaders,
            'max_timesteps': self.max_timesteps,
        }
    
    @classmethod
    def load_state_dict(cls, state_dict):
        def new_init(self, state):
            self.seed = state['SEED']
            self.batch_size = state['BATCH_SIZE']
            self.sampling_period = state['SAMPLING_PERIOD']
            self.num_dataloaders = state['NUM_DATALOADERS']
            self.max_timesteps = state['MAX_TIMESTEPS']
            self.train_set = FlightTrajectoryDataset(state['train_set'], state['sampling_period'], state['max_timesteps'])
            self.val_set = FlightTrajectoryDataset(state['val_set'], state['sampling_period'], state['max_timesteps'])
            self.test_set = FlightTrajectoryDataset(state['test_set'], state['sampling_period'], state['max_timesteps'])
        
        C = cls
        C.__init__ = new_init(state_dict)
        return C(state_dict)
    

"""

from flight_maneuvers.core import default_hparams
from flight_maneuvers.data.datamodule import FlightTrajectoryDataModule, MANEUVERS

dm = FlightTrajectoryDataModule(**default_hparams)
l = dm.train_dataloader()
it = iter(l)
b = next(it)
    


"""
