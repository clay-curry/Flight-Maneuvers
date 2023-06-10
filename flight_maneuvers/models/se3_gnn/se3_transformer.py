import torch

class SE3Transformer():
    """
    SE(3)-Transformers introduce a self-attention layer for graphs 
    that is equivariant to 3D roto-translations. It achieves this 
    by leveraging Tensor Field Networks to build attention weights 
    that are invariant and attention values that are equivariant. 
    Combining the equivariant values with the invariant weights 
    gives rise to an equivariant output. This output is normalized 
    while preserving equivariance thanks to equivariant 
    normalization layers operating on feature norms.
    """