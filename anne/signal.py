import pandas as pd
from math import inf

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

    return resampled_trajectory

def delta_timeseries(trajectory):
    """ computes the delta of a trajectory

    Args:
        trajectory: a pandas dataframe with columns t, x, y, z, vx, vy, vz, maneuver

    Returns:
        a pandas dataframe with columns t, x, y, z, dx, dy, dz, dvx, dvy, dvz, maneuver
    """
    # initialize delta_trajectory with the first row of trajectory
    delta_trajectory = trajectory.iloc[0:1]

    # compute the delta of each column
    delta_trajectory['dx'] = trajectory['x'].diff()
    delta_trajectory['dy'] = trajectory['y'].diff()
    delta_trajectory['dz'] = trajectory['z'].diff()
    delta_trajectory['dvx'] = trajectory['vx'].diff()
    delta_trajectory['dvy'] = trajectory['vy'].diff()
    delta_trajectory['dvz'] = trajectory['vz'].diff()

    # copy the maneuver column
    delta_trajectory['maneuver'] = trajectory['maneuver']

    return delta_trajectory