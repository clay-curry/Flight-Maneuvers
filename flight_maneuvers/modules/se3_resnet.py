import torch
import numpy as np
from torch import nn
from scipy import stats
import torch.nn.functional as F
# from flight_maneuvers.utils import *

from typing import Tuple
from escnn.group import *
from escnn.gspaces import *
from escnn.nn import *
from escnn.nn.modules.pointconv import R3PointConv

def make_edge_idx(n):
    return torch.stack([
            torch.hstack(
                [torch.arange(1, n, dtype=torch.long),
                 torch.arange(0, n-1, dtype=torch.long)]
            ),
            torch.hstack(
                [torch.arange(0, n-1, dtype=torch.long),
                 torch.arange(1, n, dtype=torch.long)]
            ),
        ])

class SE3_PreActResNetBlock(EquivariantModule):
    pass

class SE3_ResNetBlock(EquivariantModule):

    def __init__(self, in_type: FieldType, out_type: FieldType = None):

        super(SE3_ResNetBlock, self).__init__()
        
        self.in_type = in_type
        if out_type is None:
            self.out_type = self.in_type
        else:
            self.out_type = out_type

        gspace = self.out_type.gspace
        
        reprs = [r for r in out_type if 'gated' in r.supported_nonlinearities] 
        gates = ['gated'] * len(reprs)
        gates += ['gate'] * len(reprs)
        reprs += [gspace.trivial_repr] * len(reprs)

        self.conv1 = R3PointConv(
                    in_type, 
                    FieldType(gspace, reprs),
                    width=1,
                    n_rings=2,
                    bias=False, 
                    initialize=True)
        #self.batch_norm = NormBatchNorm(hidden_type, affine=True)
        self.nonlinearity1 = GatedNonLinearity1(FieldType(gspace, reprs), gates=gates)

        reprs = [r for r in out_type if 'gated' in r.supported_nonlinearities] 
        gates = ['gated'] * len(reprs)
        gates += ['gate'] * len(reprs)
        reprs += [gspace.trivial_repr] * len(reprs)

        self.conv2 = R3PointConv(out_type, FieldType(gspace, reprs),width=1,n_rings=2,bias=False, initialize=True)
        #self.batch_norm = NormBatchNorm(hidden_type, affine=True)
        self.nonlinearity2 = GatedNonLinearity1(FieldType(gspace, reprs), gates=gates)
        
        if self.in_type != self.out_type:
            self.skip = R3PointConv(self.in_type, self.out_type, width=10, n_rings=2,bias=False)
        else:
            self.skip = lambda x: x

    def forward(self, input: GeometricTensor, edge):
        assert input.type == self.in_type
        x = self.conv1(input, edge)
        x = self.nonlinearity1(x)
        x = self.conv2(x, edge)
        x = self.nonlinearity2(x)
        return self.skip(input, edge) + x

    def evaluate_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:

        if self.in_type != self.out_type:
            return input_shape[:1] + (self.out_type.size, ) + input_shape[2:]
        else:
            return input_shape

resnet_block_types = {
    "ResNetBlock": SE3_ResNetBlock,
    #"PreActResNetBlock": SE3_PreActResNetBlock
}


class SE3_ResNet(nn.Module):

    def __init__(self):

        super(SE3_ResNet, self).__init__()
        self._init = 'he'
        self.gs = rot3dOnR3()

        self.in_type = FieldType(self.gs, [self.gs.representations['irrep_1']])
        

        layer_types = [
            FieldType(self.gs, [self.build_representation(2)] * 3),
            FieldType(self.gs, [self.build_representation(3)] * 2),
            FieldType(self.gs, [self.build_representation(3)] * 6),
            FieldType(self.gs, [self.build_representation(3)] * 12),
            FieldType(self.gs, [self.build_representation(3)] * 8),
        ]

        blocks = [
            R3PointConv(self.in_type, layer_types[0],width=1,n_rings=2,bias=False, initialize=False)
        ]

        for i in range(len(layer_types) - 1):
            blocks.append(
                SE3_ResNetBlock(layer_types[i], layer_types[i+1])
            )
        
        self.blocks = nn.ModuleList(blocks)
        self.pool = NormPool(layer_types[-1])
        self.init()

    def init(self):
        for name, m in self.named_modules():
            if isinstance(m, R3PointConv):
                if self._init == 'he':
                    pass
                    # nn.init.generalized_he_init(m.weights.data, m.basisexpansion, cache=True)
                elif self._init == 'delta':
                    init.deltaorthonormal_init(m.weights.data, m.basisexpansion)
                elif self._init == 'rand':
                    m.weights.data[:] = torch.randn_like(m.weights)
                else:
                    raise ValueError()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                o, i = m.weight.shape
                m.weight.data[:] = torch.tensor(stats.ortho_group.rvs(max(i, o))[:o, :i])
                if m.bias is not None:
                    m.bias.data.zero_()

    def build_representation(self, K: int):
        assert K >= 0

        if K == 0:
            return [self.gs.trivial_repr]

        SO3 = self.gs.fibergroup

        polynomials = [self.gs.trivial_repr, SO3.irrep(1)]

        for k in range(2, K+1):
            polynomials.append(
                polynomials[-1].tensor(SO3.irrep(1))
            )

        return directsum(polynomials, name=f'polynomial_{K}')

    def forward(self, input: torch.Tensor):
        edge_idx = make_edge_idx(input.shape[0])
        input = GeometricTensor(input, self.in_type, input)
        
        for b in self.blocks:
            input = b(input, edge_idx)

        out = self.pool(input)
        return out


if __name__ == '__main__':

    SO3 = so3_group()
    r = SO3.irrep(1)(SO3.sample()).astype('float32')

    x = torch.randn(5, 3)
    r = torch.mm(x, torch.from_numpy(r)).shape
    
    

    # build the SE(3) equivariant model

    m.eval()

    # feed all inputs to the model
    y = m(x)

    # the outputs should be (about) the same for all transformations the model is invariant to
    print()
    print('TESTING INVARIANCE:                     ')
    print('90 degrees ROTATIONS around X axis:  ' + ('YES' if torch.allclose(y, y_x90, atol=1e-5, rtol=1e-4) else 'NO'))
    print('90 degrees ROTATIONS around Y axis:  ' + ('YES' if torch.allclose(y, y_y90, atol=1e-5, rtol=1e-4) else 'NO'))
    print('90 degrees ROTATIONS around Z axis:  ' + ('YES' if torch.allclose(y, y_z90, atol=1e-5, rtol=1e-4) else 'NO'))
    print('180 degrees ROTATIONS around Y axis: ' + ('YES' if torch.allclose(y, y_y180, atol=1e-5, rtol=1e-4) else 'NO'))
    print('REFLECTIONS on the Y axis:           ' + ('YES' if torch.allclose(y, y_fx, atol=1e-5, rtol=1e-4) else 'NO'))
    print('REFLECTIONS on the Z axis:           ' + ('YES' if torch.allclose(y, y_fy, atol=1e-5, rtol=1e-4) else 'NO'))


