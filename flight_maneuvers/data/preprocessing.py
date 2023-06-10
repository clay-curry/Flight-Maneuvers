def count_features(feature_hparams):
    """ counts the number of features in a trajectory

    Args:
        trajectory: a pandas dataframe with columns t, x, y, z, vx, vy, vz, maneuver

    Returns:
        the number of features in the trajectory
    """
    n_features = 0
    n_features += 1 if 'alt' in feature_hparams else 0
    n_features += 2 if 'pos' in feature_hparams else 0
    n_features += 3 if 'vel' in feature_hparams else 0
    n_features += 3 if 'dpos' in feature_hparams else 0
    n_features += 3 if 'dvel' in feature_hparams else 0
    return n_features

def select_features(raw_signal, feature_hparams):
    """ computes the delta of a trajectory

    Args:
        trajectory: a pandas dataframe with columns t, x, y, z, vx, vy, vz, maneuver

    Returns:
        a pandas dataframe with columns x, y, z, vx, xy, vz, dx, dy, dz, dvx, dvy, dvz, maneuver
    """
    if 'maneuver' in feature_hparams:
        trajectory['maneuver'] = raw_signal['maneuver']
        
    if 'pos' in feature_hparams:
        trajectory['x'] = raw_signal['x']
        trajectory['y'] = raw_signal['y']
    
    if 'vel' in feature_hparams:
        trajectory['vx'] = raw_signal['vx']
        trajectory['vy'] = raw_signal['vy']
        trajectory['vz'] = raw_signal['vz']

    if 'alt' in feature_hparams:
        trajectory['z'] = raw_signal['z']

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
    return trajectory.values

def get_trajectory(trajectory, sampling_period, max_length, features):
    """ loads a trajectory from a pandas dataframe

    Args:
        trajectory: a pandas dataframe with columns t, x, y, z, vx, vy, vz, maneuver
    """
    df = pd.read_csv(trajectory)
    df = sample_timeseries(df, sampling_period, max_length)
    df = select_features(df, features)
    return df

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
