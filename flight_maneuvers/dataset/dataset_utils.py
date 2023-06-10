import os
import math
import torch
import random
import numpy as np
import pandas as pd
from math import inf
from typing import List, Sized, Iterator
import lightning as L

# path pointing to a dataset of simulated flight trajectories with labeled maneuvers
TRAIN_DATA_DIR = 'maneuver_dataset'



# each training example consists of a variable-length, simulated flight trajectory with labeled maneuvers at each timestep
class FlightTrajectoryDataset(torch.utils.data.Dataset):
    
    def __init__(self, files: List[str], sampling_period, max_sample_length=math.inf):
        print("\nScenarioDataset Class: Indexing scenarios by length for more efficient sampling. This could take a second.")
        ntimesteps = {f: pd.read_csv(f)['t'].shape[0] for f in files}
        self.files = sorted(files, key=lambda x: ntimesteps[x])
        self.sampling_period = sampling_period
        self.max_sample_length = max_sample_length
        
    def __len__(self):  return len(self.files)

    def __getitem__(self, scenario_idx): 
        
        if isinstance(scenario_idx, int):
            return sample_timeseries(self.files[scenario_idx], self.sampling_period, self.max_sample_length) 
        else: 
            return [sample_timeseries(self.files[idx], self.sampling_period, self.max_sample_length) for idx in scenario_idx]

class DirectorySampler(torch.utils.data.Sampler):
    def __init__(self, data_files: Sized, replacement: bool = False, batch_size: int = 1) -> None:
        self.n_files = len(data_files)
        self.batch_size = batch_size
    
    def __iter__(self) -> Iterator[int]:
        train_samples = np.arange(self.n_files).tolist()
        while len(train_samples) > 0:
            to_take = min(self.batch_size, len(train_samples))
            select = random.randint(0, len(train_samples) - to_take)
            batch = train_samples[select:(select + to_take)]
            yield batch
            del train_samples[select:select + to_take]

    def __len__(self) -> int: return math.ceil(len(self.n_files) / self.batch_size)

    
class FlightTrajectoryDataModule:
    
    def __init__(self, num_train=1, num_valid=1, num_test=1, data_dir=TRAIN_DATA_DIR, batch_size=1, sampling_period=60, max_sample_length=256) -> None:
        super().__init__()
        self.save_hyperparameters()

        files = os.listdir(data_dir)
        assert len(files) >= num_train + num_valid + num_test
        splits = np.cumsum([num_train, num_valid, num_test])
        train_examples, valid_examples, test_examples = [f for f in np.split(random.sample(files, len(files)), splits)][:-1]
        self.train_set, self.val_set, self.test_set = (
            FlightTrajectoryDataset([os.path.join(self.hparams.data_dir, f) for f in train_examples], sampling_period=sampling_period, max_sample_length=max_sample_length),
            FlightTrajectoryDataset([os.path.join(self.hparams.data_dir, f) for f in valid_examples], sampling_period=sampling_period, max_sample_length=max_sample_length),
            FlightTrajectoryDataset([os.path.join(self.hparams.data_dir, f) for f in test_examples], sampling_period=sampling_period, max_sample_length=max_sample_length)
        )

    def tokenize_segments(self, segs):
        return segs[0]

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_set, sampler=DirectorySampler(self.train_set.files, batch_size=self.hparams.batch_size), collate_fn=self.tokenize_segments, drop_last=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_set, sampler=DirectorySampler(self.val_set.files, batch_size=self.hparams.batch_size), collate_fn=self.tokenize_segments, drop_last=True)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_set, batch_size=self.hparams.batch_size, collate_fn=self.tokenize_segments, drop_last=True)