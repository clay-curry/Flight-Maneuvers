import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch import LightningModule

from src.data_module import MANEUVERS

class SE3_ResNetBlock(nn.Module):
    pass


class SE3_PreActResNetBlock(nn.Module):
    pass


resnet_block_types = {
    "ResNetBlock": SE3_ResNetBlock,
    "PreActResNetBlock": SE3_PreActResNetBlock
}

class SE3_ResNet(LightningModule):
    pass