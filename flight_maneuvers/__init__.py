from .__about__ import *

# path pointing to a dataset of simulated flight trajectories with labeled maneuvers
DATASET_PATH = 'examples/maneuver_dataset'
# path pointing to experiment results
CHECKPOINT_PATH = 'saved_models'

__all__ = [
    "__title__",
    "__summary__",
    "__url__",
    "__version__",
    "__author__",
    "__email__",
    "__license__",
]


try:
    from flight_maneuvers import modules
    __all__ += ['modules']
except ImportError:
    pass

