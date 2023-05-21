
""" Imports core names of :mod:`flight_maneuvers.evolution.fixed_topology`.
"""

from flight_maneuvers.evolution.fixed_topology import layers

try:
    import tensorflow
    from flight_maneuvers.evolution.fixed_topology.tf_genomes import FixedTopologyGenome
except ImportError:
    pass

try:
    import torch
    from flight_maneuvers.evolution.fixed_topology.pytorch_genomes import FixedTopologyGenome
except ImportError:
    pass