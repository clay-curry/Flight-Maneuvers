
""" This module implements some activation functions.

Todo:
    Make all activation functions compatible with numpy arrays.
"""

import numpy as np


def linear(x: float) -> float:
    """ Linear activation function (simply returns the input, unchanged). """
    return x


def sigmoid(x: float,
            clip_value: int = 64) -> float:
    """ Numeric stable implementation of the sigmoid function.

    Estimated lower-bound precision with a clip value of 64: 10^(-28).
    """
    x = np.clip(x, -clip_value, clip_value)
    return 1 / (1 + np.exp(-x))


def steepened_sigmoid(x: float,
                      step: float = 4.9) -> float:
    """ Steepened version of the sigmoid function.

    The original NEAT paper used a steepened version of the sigmoid function
    with a step value of 4.9.

    "We used a modified sigmoidal transfer function,
    ϕ(x) = 1 / (1 + exp(−4.9x)), at all nodes. The steepened sigmoid allows more
    fine tuning at extreme activations. It is optimized to be close to linear
    during its steepest ascent between activations −0.5 and 0.5."
    - :cite:`stanley:ec02`
    """
    return sigmoid(x * step)
