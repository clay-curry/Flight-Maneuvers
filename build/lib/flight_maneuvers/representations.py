import numpy as np
from escnn.group import o2_group, GroupElement


def inverse_fourier_transform_O2(g: GroupElement, ft: dict):
    # the method gets in input a dictionary mapping each irrep's `id` to the corresponding Fourier Transform
    # and a group element `g`
    # The method should return the value of the function evaluated on `g`.

    G = o2_group()
    f = 0

    ########################
    # INSERT YOUR CODE HERE:
    for rho, ft_rho in ft.items():
        rho = G.irrep(*rho)
        d = rho.size
        f += np.sqrt(d) * (ft_rho * rho(g)).sum()
    ########################

    return f

G=o2_group()

irreps = [G.irrep(0, 0)] + [G.irrep(1, j) for j in range(3)]
print(irreps)
ft = {
    rho.id: np.random.randn(rho.size, rho.size)
    for rho in irreps
}


print()
print(ft)