import os
import random
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

from flight_maneuvers.data.preprocessing import *

MANEUVERS = ['takeoff', 'turn', 'line', 'orbit', 'landing'],

def tokenize_segments(self, segs):
    tokenize_maneuvers = np.vectorize(lambda id: MANEUVERS.index(id))
    # TODO
    return segs[0]

# each training example consists of a variable-length, simulated flight trajectory with labeled maneuvers at each timestep
class FlightTrajectoryDataset(Dataset):
    
    def __init__(self, files, **kwargs) -> None:
        self.kwargs = kwargs
        ntimesteps = {f: pd.read_csv(f)['t'].shape[0] for f in files}
        self.files = sorted(files, key=lambda x: ntimesteps[x])
        
    def __len__(self):  return len(self.files)

    def __getitem__(self, scenario_idx):  
        instance = sample_timeseries(
            self.files[scenario_idx], 
            self.kwargs['SAMPLING_PERIOD'],
            self.kwargs['MAX_TIMESTEPS']
        )
        states, maneuvers = get_states_maneuvers(instance)
        return to_tensors(states, maneuvers)

class FlightTrajectoryDataModule:
    
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        num_train = kwargs['TRAIN_SIZE']
        num_valid = kwargs['VAL_SIZE']
        num_test = kwargs['TEST_SIZE']
        files = os.listdir(kwargs['TRAIN_DATA_DIR'])
        assert len(files) >= num_train + num_valid + num_test
        splits = np.cumsum([num_train, num_valid, num_test])
        files = [f for f in np.split(random.sample(files, len(files)), splits)][:-1]
        self.train_set = FlightTrajectoryDataset([os.path.join(self.hparams.data_dir, f) for f in files[0]], kwargs)
        self.val_set = FlightTrajectoryDataset([os.path.join(self.hparams.data_dir, f) for f in files[1]], kwargs)
        self.test_set = FlightTrajectoryDataset([os.path.join(self.hparams.data_dir, f) for f in files[2]], kwargs)

    def train_dataloader(self):
        return DataLoader(
            self.train_set, 
            self.kwargs['BATCH_SIZE'], 
            num_workers=self.kwargs['NUM_DATALOADERS'], 
            collate_fn=tokenize_segments, 
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set, 
            self.kwargs['BATCH_SIZE'], 
            num_workers=self.kwargs['NUM_DATALOADERS'], 
            collate_fn=tokenize_segments,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set, 
            self.kwargs['BATCH_SIZE'], 
            num_workers=self.kwargs['NUM_DATALOADERS'], 
            collate_fn=tokenize_segments
        )
    
    def state_dict(self):
        return {
            'train_set': self.train_set.files,
            'val_set': self.val_set.files,
            'test_set': self.test_set.files,
            'kwargs': self.kwargs
        }
    
    @classmethod
    def load_state_dict(cls, state_dict):
        def new_init(self, state):
            self.kwargs = state['kwargs']
            self.train_set = FlightTrajectoryDataset(state['train_set'], state['kwargs'])
            self.val_set = FlightTrajectoryDataset(state['val_set'], state['kwargs'])
            self.test_set = FlightTrajectoryDataset(state['test_set'], state['kwargs'])
        
        C = cls
        C.__init__ = new_init(state_dict)
        return C(state_dict)