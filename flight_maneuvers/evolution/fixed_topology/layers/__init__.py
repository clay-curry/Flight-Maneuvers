
""" Neural network layers to be used with `NEvoPy's` fixed-topology
neuroevolutionary algorithms.
"""

from nevopy.fixed_topology.layers import mating

from nevopy.fixed_topology.layers.base_layer import BaseLayer
from nevopy.fixed_topology.layers.base_layer import IncompatibleLayersError

from nevopy.fixed_topology.layers.tf_layers import TensorFlowLayer
from nevopy.fixed_topology.layers.tf_layers import TFConv2DLayer
from nevopy.fixed_topology.layers.tf_layers import TFDenseLayer
from nevopy.fixed_topology.layers.tf_layers import TFFlattenLayer
from nevopy.fixed_topology.layers.tf_layers import TFMaxPool2DLayer
