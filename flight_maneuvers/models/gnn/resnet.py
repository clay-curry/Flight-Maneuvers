import torch
import torch.nn as nn
from itertools import tee
from flight_maneuvers.data.features import count_features

class ResNetBlock(nn.Module):

    def __init__(self, c_in, k_size, act_fn, c_out=-1):
        """
        Inputs:
            c_in - Number of input features
            act_fn - Activation class constructor (e.g. nn.ReLU)
            subsample - If True, we want to apply a stride inside the block and reduce the output shape by 2 in height and width
            c_out - Number of output features. Note that this is only relevant if subsample is True, as otherwise, c_out = c_in
        """
        super().__init__()     
        # linear
        self.proj = nn.Conv1d(c_in, c_out, kernel_size=1, padding='same', bias=False)
        # residual
        self.net = nn.Sequential(
            nn.Conv1d(c_in, c_out, kernel_size=k_size, padding='same', bias=False),  # No bias needed as the Batch Norm handles it
            nn.BatchNorm1d(c_out),
            act_fn(),
            nn.Conv1d(c_out, c_out, kernel_size=k_size, padding='same', bias=False),
            nn.BatchNorm1d(c_out)
        )

        self.act_fn = act_fn()

    def forward(self, x):
        z = self.net(x)
        out = z + self.proj(x)
        return self.act_fn(out)
        

class PreActResNetBlock(nn.Module):
    def __init__(self, c_in, k_size, act_fn, c_out=-1):
        """
        Inputs:
            c_in - Number of input features
            act_fn - Activation class constructor (e.g. nn.ReLU)
            subsample - If True, we want to apply a stride inside the block and reduce the output shape by 2 in height and width
            c_out - Number of output features. Note that this is only relevant if subsample is True, as otherwise, c_out = c_in
        """
        super().__init__()
        self.proj = nn.Conv1d(c_in, c_out, kernel_size=1, padding='same', bias=False)
        self.act_fn = act_fn()
        # Network representing F
        self.net = nn.Sequential(
            nn.BatchNorm1d(c_in),
            act_fn(),
            nn.Conv1d(c_in, c_out, kernel_size=k_size, padding='same', bias=False),
            nn.BatchNorm1d(c_out),
            act_fn(),
            nn.Conv1d(c_out, c_out, kernel_size=k_size, padding='same', bias=False)
        )

    def forward(self, x):
        z = self.net(x)
        return z + self.proj(x)

resnet_block_types = {
    "ResNetBlock": ResNetBlock,
    "PreActResNetBlock": PreActResNetBlock
}

act_fn_by_name = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "leakyrelu": nn.LeakyReLU,
    "gelu": nn.GELU
}

class ResNet(nn.Module):

    def __init__(self, 
                c_hidden=[16,32,64], 
                kernel_size=[3,3,3],
                act_fn_name="relu", 
                block_name="ResNetBlock", 
                num_maneuvers=10, 
                **kwargs
        ):
        """
        Inputs:
            num_classes - Number of classification outputs (10 for CIFAR10)
            num_blocks - List with the number of ResNet blocks to use. The first block of each group uses downsampling, except the first.
            c_hidden - List with the hidden dimensionalities in the different blocks. Usually multiplied by 2 the deeper we go.
            act_fn_name - Name of the activation function to use, looked up in "act_fn_by_name"
            block_name - Name of the ResNet block, looked up in "resnet_blocks_by_name"
        """
        super().__init__()
        assert block_name in resnet_block_types
        self.act_fn_name = act_fn_name

        state_dim = count_features(kwargs['feature_hparams'])
                
        # A first convolution on the original image to scale up the channel size
        self.input_net = nn.Sequential(
                nn.Conv1d(state_dim, c_hidden[0], kernel_size=kernel_size, padding="same", bias=False)
            ) if resnet_block_types[block_name] == PreActResNetBlock else nn.Sequential(
                nn.Conv1d(state_dim, c_hidden[0], kernel_size=kernel_size, padding="same", bias=False),
                nn.BatchNorm1d(c_hidden[0]),
                act_fn_by_name[act_fn_name]()
            )
        
        # Creating the ResNet blocks
        self.blocks = nn.Sequential(*[
            resnet_block_types[block_name](
                c_in=c_in,
                act_fn=act_fn_by_name[act_fn_name],
                k_size=kernel_size,
                c_out=c_out
            ) for block_idx, (c_in, c_out) in pairwise(c_hidden)
        ])

        # Mapping to classification output
        self.output_net = nn.Sequential(
            nn.Linear(c_hidden[-1], c_hidden[-1]),
            act_fn_by_name[act_fn_name](),
            nn.Linear(c_hidden[-1], num_maneuvers)
        )
        
        self._init_params()


    def _init_params(self):
        # Based on our discussion in Tutorial 4, we should initialize the convolutions according to the activation function
        # Fan-out focuses on the gradient distribution, and is commonly used in ResNets
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = torch.transpose(x, -1, -2).unsqueeze(0).contiguous()        
        x = self.input_net(x)
        x = self.blocks(x).transpose(-1, -2).squeeze(0)
        x = self.output_net(x)
        return x