import unittest
from unittest import TestCase

from flight_maneuvers.utils import *
from flight_maneuvers.modules.se3_resnet import SE3_ResNet
from flight_maneuvers.data.datamodule import MANEUVERS, FlightTrajectoryDataModule

class TestResnet(TestCase):
    
    def test_se3resnet(self):
        model_params = {
            "num_maneuvers": len(MANEUVERS),
            "num_blocks": [3, 3, 3],
            "c_hidden": [16, 32, 64],
            "kernel_size": [3, 3, 3],
            "act_fn_name": "relu",
        }
        
        model = SE3_ResNet(**model_params)
        self.assertIsNotNone(model)

    def test_se3resnet_invariance(self):
        x = torch.randn(5, 3)

        # the outputs should be (about) the same for all transformations the model is invariant to
        print()
        print('TESTING INVARIANCE:                     ')
        print('90 degrees ROTATIONS around X axis:  ' + ('YES' if torch.allclose(y, y_x90, atol=1e-5, rtol=1e-4) else 'NO'))
        print('90 degrees ROTATIONS around Y axis:  ' + ('YES' if torch.allclose(y, y_y90, atol=1e-5, rtol=1e-4) else 'NO'))
        print('90 degrees ROTATIONS around Z axis:  ' + ('YES' if torch.allclose(y, y_z90, atol=1e-5, rtol=1e-4) else 'NO'))
        print('180 degrees ROTATIONS around Y axis: ' + ('YES' if torch.allclose(y, y_y180, atol=1e-5, rtol=1e-4) else 'NO'))
        print('REFLECTIONS on the Y axis:           ' + ('YES' if torch.allclose(y, y_fx, atol=1e-5, rtol=1e-4) else 'NO'))
        print('REFLECTIONS on the Z axis:           ' + ('YES' if torch.allclose(y, y_fy, atol=1e-5, rtol=1e-4) else 'NO'))



    