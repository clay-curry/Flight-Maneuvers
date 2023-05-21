
""" Neural network layers to be used with `flight_maneuvers.evolution's` fixed-topology
neuroevolutionary algorithms.
"""

from flight_maneuvers.evolution.fixed_topology.layers import mating

from flight_maneuvers.evolution.fixed_topology.layers.base_layer import BaseLayer
from flight_maneuvers.evolution.fixed_topology.layers.base_layer import IncompatibleLayersError

try:
    import tensorflow 
    from flight_maneuvers.evolution.fixed_topology.layers.tf_layers import TensorFlowLayer
    from flight_maneuvers.evolution.fixed_topology.layers.tf_layers import TFConv2DLayer
    from flight_maneuvers.evolution.fixed_topology.layers.tf_layers import TFDenseLayer
    from flight_maneuvers.evolution.fixed_topology.layers.tf_layers import TFFlattenLayer
    from flight_maneuvers.evolution.fixed_topology.layers.tf_layers import TFMaxPool2DLayer
except ImportError:
    pass

try:
    import torch
    from flight_maneuvers.evolution.fixed_topology.layers.torch_layers import TorchLayer
    from flight_maneuvers.evolution.fixed_topology.layers.torch_layers import TorchConv2DLayer
    from flight_maneuvers.evolution.fixed_topology.layers.torch_layers import TorchDenseLayer
    from flight_maneuvers.evolution.fixed_topology.layers.torch_layers import TorchFlattenLayer
    from flight_maneuvers.evolution.fixed_topology.layers.torch_layers import TorchMaxPool2DLayer
except ImportError:
    pass