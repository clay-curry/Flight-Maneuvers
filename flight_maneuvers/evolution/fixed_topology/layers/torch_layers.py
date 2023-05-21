
""" Implements subclasses of :class:`.BaseLayer` that wrap Pytorch layers.
"""

import torch
import numpy as np

from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

from flight_maneuvers.evolution.base_genome import InvalidInputError
from flight_maneuvers.evolution.config import GeneticAlgorithmConfig
from flight_maneuvers.evolution.fixed_topology.layers import mating
from flight_maneuvers.evolution.fixed_topology.layers.base_layer import BaseLayer
from flight_maneuvers.evolution.fixed_topology.layers.base_layer import IncompatibleLayersError


class PytorchLayer(BaseLayer):
    """ Wraps a `PytorchLayer` layer.

    This class wraps a `PytorchLayer` layer, making it compatible with `flight_maneuvers.evolution's`
    neuroevolutionary algorithms. It handles the mutation and reproduction of
    the `PytorchLayer` layer.

    In most cases, there is no need to create subclasses of this class. Doing
    that to frequently used types of layers, however, may be desirable, since it
    makes using those types of layers easier (see :class:`.Conv2DLayer` and
    :class:`.DenseLayer` as examples).

    When inheriting this class, you'll usually do something like this:

        .. code-block:: python

            class MyTorchLayer(PytorchLayer):
                def __init__(self,
                             arg1, arg2,
                             activation="relu",
                             mating_func=mating.exchange_units_mating,
                             config=None,
                             input_shape=None,
                             mutable=True,
                             **pt_kwargs: Dict[str, Any]):
                    super().__init__(
                        layer_type=torch.nn.SomeKerasLayer,
                        **{k: v for k, v in locals().items()
                           if k not in ["self", "pt_kwargs", "__class__"]},
                        **pt_kwargs,
                    )

    Args:
        layer_type (Union[str, Type[torch.nn.Layer]]): A reference to the
            `Pytorch's` class that represents the layer
            (:py:class:`torch.nn.Dense`, for example). If it's a string,
            the appropriate type will be inferred (note that it must be listed
            in :attr:`.PytorchLayer.KERAS_LAYERS`).
        mating_func (Optional[Callable[[BaseLayer, BaseLayer], BaseLayer]]):
            Function that mates (sexual reproduction) two layers. It should
            receive two layers as input and return a new layer (the offspring).
            You can use one of the pre-built mating functions (see
            :mod:`.fixed_topology.layers.mating`) or implement your own. If the
            layer is immutable, this parameter should receive `None` as
            argument.
        config (Optional[FixedTopologyConfig]): Settings being used in the
            current evolutionary session. If `None`, a config object must be
            assigned to the layer later on, before calling the methods that
            require it.
        input_shape (Optional[Tuple[int, ...]]): Shape of the data that will be
            processed by the layer. If `None`, an input shape for the layer must
            be manually specified later or be inferred from an input sample.
        mutable (Optional[bool]): Whether or not the layer can have its weights
            changed (mutation).
        **pt_kwargs: Named arguments to be passed to the constructor of the
            `Pytorch` layer.
    """

    MODULES = {
        "conv2D": torch.nn.Conv2d,
        "dense": torch.nn.Linear,
        "flatten": torch.nn.Flatten,
        "maxpool2D": torch.nn.MaxPool2d,
    }

    def __init__(self,
                 layer_type: Union[str, Type[torch.nn.Module]],
                 mating_func: Optional[
                     Callable[[BaseLayer, BaseLayer], BaseLayer]
                 ] = mating.exchange_units_mating,
                 config: Optional[GeneticAlgorithmConfig] = None,
                 input_shape: Optional[Tuple[int, ...]] = None,
                 mutable: Optional[bool] = True,
                 **pt_kwargs) -> None:
        
        super().__init__(config, input_shape, mutable)
        self._layer_type = (layer_type if not isinstance(layer_type, str) else PytorchLayer.MODULES[layer_type])
        self._pt_layer_kwargs = pt_kwargs
        self.mating_func = mating_func

        self._pt_layer = self._layer_type(**self._pt_layer_kwargs)

    @property
    def _(self) -> torch.nn.Module:
        """
        The `torch.nn.Module` used internally.
        """
        return self._modules

    @property
    def weights(self) -> List[np.ndarray]:
        """ The current weight matrices of the layer.

        Wrapper for :meth:`torch.nn.Layer.get_weights`.

        The weights of a layer represent the state of the layer. This property
        returns the weight values associated with this layer as a list of Numpy
        arrays. In most cases, it's a list containing the weights of the layer's
        connections and the bias values (one for each neuron, generally).
        """
        return [w.numpy() for w in self.pt_layer.weights]

    @weights.setter
    def weights(self, new_weights: List[np.ndarray]) -> None:
        """ Wrapper for :meth:`torch.nn.Layer.set_weights()`. """
        self.pt_layer.set_weights(new_weights)

    def build(self, input_shape: Tuple[int, ...]) -> None:
        """ Wrapper for :meth:`torch.nn.Layer.build()`. """
        self.pt_layer.build(input_shape=input_shape)
        self._input_shape = input_shape

    def process(self, x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        try:
            result = self.pt_layer(x)
            if self._input_shape is None:
                self._input_shape = x.shape
            return result
        except ValueError as e:
            raise InvalidInputError(
                "The given input's shape doesn't match the shape expected by "
                f"the layer! Pytorch's error message: {str(e)}"
            ) from e

    def _new_instance(self):
        """ Returns a new instance of the layer.

        The new instance doesn't inherit the current layer's weights - a new set
        of weights is initialized.
        """
        return PytorchLayer(layer_type=self._layer_type,
                               mating_func=self.mating_func,
                               config=self.config,
                               input_shape=self._input_shape,
                               mutable=self.mutable,
                               **self._pt_layer_kwargs)

    def random_copy(self) -> "PytorchLayer":
        if not self.mutable:
            return self.deep_copy()
        return self._new_instance()

    def deep_copy(self) -> "PytorchLayer":
        new_layer = self._new_instance()
        new_layer.weights = self.weights
        return new_layer

    def mutate_weights(self,
                       # pylint: disable=invalid-name
                       _test_info: Dict = None,
    ) -> None:
        """ Randomly mutates the weights of the layer's connections.

        Each weight has a chance to be perturbed by a predefined amount or to be
        reset. The probabilities are obtained from the settings of the current
        evolutionary session.

        If the layer is immutable, nothing happens (the layer's weights remain
        unchanged).
        """
        if not self.mutable:
            return

        assert self.config is not None
        if self.input_shape is None:
            raise RuntimeError("Attempt to mutate the weights of a layer that "
                               "didn't have its weight and bias matrices "
                               "initialized!")

        new_weights = []
        for i, w in enumerate(self.weights):
            old_shape = w.shape

            # Mutating weights:
            num_mutate = np.random.binomial(w.size,
                                            self.config.weight_mutation_chance)
            if num_mutate > 0:
                w_perturbation = np.random.uniform(
                    low=1 - self.config.weight_perturbation_pc,
                    high=1 + self.config.weight_perturbation_pc,
                    size=num_mutate,
                )
                mutate_idx = np.random.choice(range(w.size),
                                              size=num_mutate,
                                              replace=False)
                w.flat[mutate_idx] = np.multiply(w.flat[mutate_idx],
                                                 w_perturbation)

            # Resetting weights:
            num_reset = np.random.binomial(w.size,
                                           self.config.weight_reset_chance)
            if num_reset > 0:
                reset_idx = np.random.choice(range(w.size),
                                             size=num_reset,
                                             replace=False)
                w.flat[reset_idx] = np.random.uniform(
                    low=self.config.new_weight_interval[0],
                    high=self.config.new_weight_interval[1],
                    size=num_reset,
                )

            # Saving weight matrix:
            assert w.shape == old_shape
            new_weights.append(w)

            # Test/debug info:
            if _test_info is not None:
                # noinspection PyUnboundLocalVariable
                _test_info[f"w{i}_perturbation"] = (w_perturbation
                                                    if num_mutate > 0
                                                    else np.array([]))
                # noinspection PyUnboundLocalVariable
                _test_info[f"w{i}_mutate_idx"] = (mutate_idx if num_mutate > 0
                                                  else np.array([]))
                # noinspection PyUnboundLocalVariable
                _test_info[f"w{i}_reset_idx"] = (reset_idx if num_reset > 0
                                                 else np.array([]))

        # setting new weights and biases
        self.weights = new_weights

    def mate(self, other: "PytorchLayer") -> "PytorchLayer":
        if self.mutable != other.mutable:
            raise IncompatibleLayersError("Attempt to mate an immutable "
                                          "layer with a mutable layer!")

        if other == self or not self.mutable:
            return self.deep_copy()

        return self.mating_func(self, other)  # type: ignore


class Conv2DLayer(PytorchLayer):
    """ Wraps a `Pytorch` 2D convolution layer.

    This is a simple wrapper for `torch.nn.Conv2D`
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: Tuple[int, int], 
                 strides: Tuple[int, int] = (1, 1), padding: str = "same", activation="relu",
                 bias: bool = True, padding_mode: str = 'zeros',  device=None, dtype=None,
                 mating_func: Optional[ Callable[[BaseLayer, BaseLayer], BaseLayer]] = mating.exchange_units_mating,
                 config: Optional[GeneticAlgorithmConfig] = None,
                 input_shape: Optional[Tuple[int, ...]] = None,
                 mutable: Optional[bool] = True,  **pt_kwargs: Dict[str, Any]
            ) -> None:

        super().__init__(
            layer_type=torch.nn.Conv2d,
            **{k: v for k, v in locals().items()
               if k not in ["self", "pt_kwargs", "__class__"]},
            **pt_kwargs,
        )


class Linear(PytorchLayer):
    """ Wraps a `Pytorch` dense layer.

    This is a simple wrapper for `torch.nn.Dense
    <https://www.Pytorch.org/api_docs/python/tf/keras/layers/Dense>`_.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                device=None, dtype=None, activation=None, mutable: Optional[bool] = True,
                mating_func: Optional[Callable[[BaseLayer, BaseLayer], BaseLayer]] = mating.exchange_weights_mating,
                input_shape: Optional[Tuple[int, ...]] = None,
                config: Optional[GeneticAlgorithmConfig] = None, **pt_kwargs: Dict[str, Any]
            ) -> None:
        
        super().__init__(
            layer_type=torch.nn.Linear,
            **{k: v for k, v in locals().items() if k not in ["self", "pt_kwargs", "__class__"]},
            **pt_kwargs
        )


class FlattenLayer(PytorchLayer):
    """ Wraps a `Pytorch` flatten layer.

    This is a simple wrapper for `torch.nn.Flatten
    <https://www.Pytorch.org/api_docs/python/tf/keras/layers/Flatten>`_.
    """

    def __init__(self, start_dim: int = 1, end_dim: int = -1, 
                 mating_func: Optional[ Callable[[BaseLayer, BaseLayer], BaseLayer] ] = None,
                 config: Optional[GeneticAlgorithmConfig] = None,
                 input_shape: Optional[Tuple[int, ...]] = None,
                 mutable: Optional[bool] = False, **pt_kwargs: Dict[str, Any]
            ) -> None:
        
        super().__init__(
            layer_type=torch.nn.Flatten,
            **{k: v for k, v in locals().items()
               if k not in ["self", "pt_kwargs", "__class__"]},
            **pt_kwargs,
        )


class MaxPool2DLayer(PytorchLayer):
    """ Wraps a `Pytorch` 2D max pooling layer.

    This is a simple wrapper for `torch.nn.MaxPool2D
    <https://www.Pytorch.org/api_docs/python/tf/keras/layers/Flatten>`_.
    """

    def __init__(self, pool_size: Tuple[int, int] = (2, 2),
                 strides: Optional[Tuple[int, int]] = None, padding: str = "valid",
                 mating_func: Optional[ Callable[[BaseLayer, BaseLayer], BaseLayer]] = None,
                 config: Optional[GeneticAlgorithmConfig] = None,
                 input_shape: Optional[Tuple[int, ...]] = None,
                 mutable: Optional[bool] = False, **pt_kwargs: Dict[str, Any]
            ) -> None:
    
        kernel_size = pool_size
        stride = strides
        super().__init__(
            layer_type=torch.nn.MaxPool2d,
            **{k: v for k, v in locals().items()
               if k not in ["self", "pt_kwargs", "__class__"]},
            **pt_kwargs,
        )
