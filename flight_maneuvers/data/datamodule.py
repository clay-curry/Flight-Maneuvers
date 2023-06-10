import os
import random
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

from flight_maneuvers.data.timeseries_sampler import open_trajectory


def tokenize_segments(self, segs):
    tokenize_maneuvers = np.vectorize(lambda id: MANEUVERS.index(id))
    # TODO
    return segs[0]

# each training example consists of a variable-length, simulated flight trajectory with labeled maneuvers at each timestep
class FlightTrajectoryDataset(Dataset):
    
    def __init__(self, files, sampling_period, max_timesteps, features) -> None:
        self.features = features
        self.sampling_period = sampling_period
        self.max_timesteps = max_timesteps
        ntimesteps = {f: pd.read_csv(f)['t'].shape[0] for f in files}
        self.files = sorted(files, key=lambda x: ntimesteps[x])
        
    def __len__(self):  return len(self.files)

    def __getitem__(self, scenario_idx):  return [open_trajectory(self.files[idx], self.sampling_period, self.max_timesteps, self.features) for idx in scenario_idx]


class FlightTrajectoryDataModule:
    
    def __init__(self, num_train, num_valid, num_test, batch_size, sampling_period=SAMPLING_PERIOD, max_timesteps=MAX_TIMESTEPS, data_dir=TRAIN_DATA_DIR, num_workers=NUM_WORKERS) -> None:
        files = os.listdir(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        assert len(files) >= num_train + num_valid + num_test
        splits = np.cumsum([num_train, num_valid, num_test])
        files = [f for f in np.split(random.sample(files, len(files)), splits)][:-1]
        self.train_set = FlightTrajectoryDataset([os.path.join(self.hparams.data_dir, f) for f in files[0]], sampling_period, max_timesteps)
        self.val_set = FlightTrajectoryDataset([os.path.join(self.hparams.data_dir, f) for f in files[1]], sampling_period, max_timesteps)
        self.test_set = FlightTrajectoryDataset([os.path.join(self.hparams.data_dir, f) for f in files[2]], sampling_period, max_timesteps)

    def train_dataloader(self):
        return DataLoader(self.train_set, self.batch_size, collate_fn=tokenize_segments, num_workers=self.num_workers, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, self.batch_size, collate_fn=tokenize_segments, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, self.batch_size, collate_fn=tokenize_segments, num_workers=self.num_workers)