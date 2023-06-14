import torch
import pandas as pd

def make_edge_idx(n):
    return torch.stack([
            torch.hstack(
                [torch.arange(1, n, dtype=torch.long),
                 torch.arange(0, n-1, dtype=torch.long)]
            ),
            torch.hstack(
                [torch.arange(0, n-1, dtype=torch.long),
                 torch.arange(1, n, dtype=torch.long)]
            ),
        ])

def sample_timeseries(trajectory, sampling_period, max_length):
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


def get_states_maneuvers(raw_signal):
    """ computes the delta of a trajectory

    Args:
        trajectory: a pandas dataframe with columns t,  z, vx, vy, vz, maneuver

    Returns:
        a pandas dataframe with columns z, vx, xy, vz, dx, dy, dz, dvx, dvy, dvz, maneuver
    """
    trajectory = raw_signal[['z', 'vx', 'vy', 'vz']]
    trajectory = trajectory.assign(dx=raw_signal['x'].diff())
    trajectory = trajectory.assign(dy=raw_signal['y'].diff())
    trajectory = trajectory.assign(dz=raw_signal['z'].diff())
    trajectory['dx'].iloc[0] = trajectory['dy'].iloc[0] = trajectory['dz'].iloc[0] = 0
    trajectory = trajectory.assign(dvx=raw_signal['vx'].diff())
    trajectory = trajectory.assign(dvy=raw_signal['vy'].diff())
    trajectory = trajectory.assign(dvz=raw_signal['vz'].diff())
    trajectory['dvx'].iloc[0] = trajectory['dvy'] = trajectory['dvz'] = 0
    # initialize delta_trajectory with the first row of trajectory
    return trajectory, raw_signal['maneuver']

def to_tensors(trajectory, maneuver):
    """ converts a trajectory to a torch tensor

    Args:
        trajectory: a pandas dataframe with columns x, y, z, vx, xy, vz, dx, dy, dz, dvx, dvy, dvz
        maneuver: a pandas series with categorical maneuvers

    Returns:
        a torch tensor with shape (n_timesteps, n_features)
    """
    return torch.tensor(trajectory.values, dtype=torch.float32), torch.tensor(maneuver.values, dtype=torch.long)