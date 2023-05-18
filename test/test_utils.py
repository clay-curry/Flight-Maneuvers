import unittest
from unittest import TestCase

from flight_maneuvers.modules.resnet import ResNet
from flight_maneuvers.modules import FlightManeuverModule
from flight_maneuvers.data_module import MANEUVERS, FlightTrajectoryDataModule

class TestResnet(TestCase):
    
    def test_resnet_2blocks(self):
        model_params = {
            "state_dim": 12,
            "num_maneuvers": len(MANEUVERS),
            "num_blocks": [3, 3, 3],
            "c_hidden": [16, 32, 64],
            "kernel_size": [3, 3, 3],
            "act_fn_name": "relu",
        }
        model = FlightManeuverModule(ResNet, model_params)

        assert model is not None            



if __name__ == '__main__':
    unittest.main()
