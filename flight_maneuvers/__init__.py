from .__about__ import *



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

from flight_maneuvers.evolution import *

