from flight_maneuvers.modules.resnet import ResNet
from flight_maneuvers.utils import *

model = train(ResNet, {
    "state_dim": 12,
    "num_maneuvers": len(MANEUVERS),
    "num_blocks": [3] * 30,
    "c_hidden": [16] * 30,
    "kernel_size": [3] * 30,
    "act_fn_name": "relu",
    "block_name": "PreActResNetBlock"
})