class SE2Transformer():
    """
    SE(2)-Transformers introduce a self-attention layer for graphs 
    that is equivariant to the group of 2D roto-translations leaving 
    the direction of the Z axis unchanged. It achieves this 
    by leveraging Tensor Field Networks to build attention weights 
    that are invariant and attention values that are equivariant. 
    Combining the equivariant values with the invariant weights 
    gives rise to an equivariant output. This output is normalized 
    while preserving equivariance thanks to equivariant 
    normalization layers operating on feature norms.
    """