import unittest
from unittest import TestCase

from flight_maneuvers.utils import *
from flight_maneuvers.modules.resnet import ResNet
from flight_maneuvers.data_module import MANEUVERS, FlightTrajectoryDataModule

class TestResnet(TestCase):
    
    def test_resnet(self):
        model_params = {
            "state_dim": 12,
            "num_maneuvers": len(MANEUVERS),
            "num_blocks": [3, 3, 3],
            "c_hidden": [16, 32, 64],
            "kernel_size": [3, 3, 3],
            "act_fn_name": "relu",
        }
        model = ResNet(**model_params)
        self.assertIsNotNone(model)

    def test_resnet_forward(self):
        model_params = {
            "state_dim": 12,
            "num_maneuvers": len(MANEUVERS),
            "num_blocks": [3, 3, 3],
            "c_hidden": [16, 32, 64],
            "kernel_size": [3, 3, 3],
            "act_fn_name": "relu",
        }
        model = ResNet(**model_params)
        for batch in FlightTrajectoryDataModule().train_dataloader():
            x, maneuver = preprocess_trajectory(batch[0])
            out = model.forward(x)
            self.assertEqual(out.shape, (x.shape[0], model_params["num_maneuvers"]))

    def test_train_resnet(self):
        model = train(ResNet, {
            "state_dim": 12,
            "num_maneuvers": len(MANEUVERS),
            "num_blocks": [3] * 30,
            "c_hidden": [16] * 30,
            "kernel_size": [3] * 30,
            "act_fn_name": "relu",
        })
        self.assertIsNotNone( model )
        


if __name__ == '__main__':
    unittest.main()
