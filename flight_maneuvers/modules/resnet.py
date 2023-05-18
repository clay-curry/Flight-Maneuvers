import torch
import torch.nn as nn
import torch.nn.functional as F

from flight_maneuvers.data_module import MANEUVERS

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

                
        self.proj = nn.Conv1d(c_in, c_out, kernel_size=1, padding='same', bias=False)
        # Network representing F
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
        out = self.act_fn(out)
        return out


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

    def __init__(self, state_dim=12, num_blocks=[3,3,3], kernel_size=[3,3,3], c_hidden=[16,32,64], act_fn_name="relu", block_name="ResNetBlock", num_maneuvers=10, **kwargs):
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
        self.state_dim=state_dim
        self.c_hidden=c_hidden
        self.kernel_size=kernel_size
        self.num_blocks=num_blocks
        self.act_fn_name=act_fn_name
        self.act_fn=act_fn_by_name[act_fn_name]
        self.block_class=resnet_block_types[block_name]
        self.num_maneuvers=num_maneuvers
        self._create_network()
        self._init_params()

    def _create_network(self):
        c_hidden = self.c_hidden
        kernel_size = self.kernel_size
        # A first convolution on the original image to scale up the channel size
        if self.block_class == PreActResNetBlock: # => Don't apply non-linearity on output
            self.input_net = nn.Sequential(
                nn.Conv1d(self.state_dim, c_hidden[0], kernel_size=kernel_size[0], padding="same", bias=False)
            )
        else:
            self.input_net = nn.Sequential(
                nn.Conv1d(self.state_dim, c_hidden[0], kernel_size=kernel_size[0], padding="same", bias=False),
                nn.BatchNorm1d(c_hidden[0]),
                self.act_fn()
            )
        c_hidden = [c_hidden[0]] + c_hidden
        
        # Creating the ResNet blocks
        blocks = []
        for block_idx, block_count in enumerate(self.num_blocks):
            blocks.append(
                    self.block_class(
                        c_in=c_hidden[block_idx],
                        act_fn=self.act_fn,
                        k_size=kernel_size[block_idx],
                        c_out=c_hidden[block_idx+1])
                )
            for bc in range(block_count - 1):
                blocks.append(
                    self.block_class(
                        c_in=c_hidden[block_idx+1],
                        act_fn=self.act_fn,
                        k_size=kernel_size[block_idx],
                        c_out=c_hidden[block_idx+1])
                )

        self.blocks = blocks

        # Mapping to classification output
        self.output_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c_hidden[-1], self.num_maneuvers)
        )

    def _init_params(self):
        # Based on our discussion in Tutorial 4, we should initialize the convolutions according to the activation function
        # Fan-out focuses on the gradient distribution, and is commonly used in ResNets
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=self.act_fn_name)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = torch.transpose(x, -1, -2).unsqueeze(0).contiguous()
        
        x = self.input_net(x)
        for block in self.blocks:
            x = block(x)
        x = torch.transpose(x, -1, -2).squeeze(0)

        x = self.output_net(x)
        return x