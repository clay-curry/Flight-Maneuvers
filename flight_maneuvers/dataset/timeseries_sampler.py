import torch
import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

from flight_maneuvers.data.dataset_utils import MANEUVERS
from flight_maneuvers.modules.resnet import ResNet, PreActResNetBlock
from flight_maneuvers.data.dataset_utils import FlightTrajectoryDataModule

tokenize_maneuvers = np.vectorize(lambda id: MANEUVERS.index(id))


def preprocess_trajectory(raw_signal, feature_hparams):
    """ computes the delta of a trajectory

    Args:
        trajectory: a pandas dataframe with columns t, x, y, z, vx, vy, vz, maneuver

    Returns:
        a pandas dataframe with columns x, y, z, vx, xy, vz, dx, dy, dz, dvx, dvy, dvz, maneuver
    """
    maneuvers = tokenize_maneuvers(raw_signal['maneuver'])
    trajectory = raw_signal[['z']]
    
    if 'pos' in feature_hparams:
        trajectory['x'] = raw_signal['x']
        trajectory['y'] = raw_signal['y']
    
    if 'vel' in feature_hparams:
        trajectory['vx'] = raw_signal['vx']
        trajectory['vy'] = raw_signal['vy']
        trajectory['vz'] = raw_signal['vz']

    if 'dpos' in feature_hparams:
        trajectory = trajectory.assign(dx=raw_signal['x'].diff())
        trajectory = trajectory.assign(dy=raw_signal['y'].diff())
        trajectory = trajectory.assign(dz=raw_signal['z'].diff())
        trajectory['dx'].iloc[0] = trajectory['dy'].iloc[0] = trajectory['dz'].iloc[0] = 0

    if 'dvel' in feature_hparams:
        trajectory = trajectory.assign(dvx=raw_signal['vx'].diff())
        trajectory = trajectory.assign(dvy=raw_signal['vy'].diff())
        trajectory = trajectory.assign(dvz=raw_signal['vz'].diff())
        trajectory['dvx'].iloc[0] = trajectory['dvy'] = trajectory['dvz'] = 0
    # initialize delta_trajectory with the first row of trajectory
    return torch.tensor(trajectory.values, dtype=torch.float32), torch.tensor(maneuvers)

def postprocess_joint(joint_dist):
    """ casts joint distribution to a dataframe, and appends the predicted maneuver
    """
    joint_df = pd.DataFrame({
            'takeoff': joint_dist[:, 0],
            'turn': joint_dist[:, 1],
            'line': joint_dist[:, 2],
            'orbit': joint_dist[:, 3],
            'landing': joint_dist[:, 4]
        })
    joint_df['maneuver'] = joint_df.idxmax(axis="columns")
    return joint_df

def sample_timeseries(trajectory, sampling_period=1, max_length=inf):
    """ loads and resamples a trajectory to a fixed sampling period and maximum length

    Args:
        trajectory: path to a pandas dataframe with columns t, x, y, z, vx, vy, vz, maneuver
        sampling_period: the desired sampling period in seconds
        max_length: the maximum length of the resampled trajectory

    Returns:
        a pandas dataframe with columns t, x, y, z, vx, vy, vz, maneuver
    """
    # load the trajectory and convert the 't' column to a DatetimeIndex
    trajectory = pd.read_csv(trajectory)

    # initialize resampled_trajectory with the first row of trajectory
    resampled_trajectory = trajectory.iloc[::sampling_period]

    # trim the trajectory to the desired maximum length
    if len(resampled_trajectory) > max_length:
        resampled_trajectory = resampled_trajectory.iloc[:max_length]

    resampled_trajectory = resampled_trajectory.reset_index(drop=True)
    return resampled_trajectory
