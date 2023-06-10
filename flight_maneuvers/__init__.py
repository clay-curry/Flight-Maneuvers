# path pointing to a dataset of experiment logs, consisting of checkpoints and tensorboard logs
LOG_FILE_DIR = 'logs/eval'

# categorical maneuvers assigned to the projectile at each timestep
MANEUVERS = ['takeoff', 'turn', 'line', 'orbit', 'landing']


__all__ = [
    "TRAIN_DATA_DIR",
    "LOG_FILE_DIR",
    "MANEUVERS",
]

try:
    from flight_maneuvers import modules
    __all__ += ['modules']
except ImportError:
    pass

try:
    from flight_maneuvers import evolution
    __all__ += ['evolution']
except ImportError:
    pass