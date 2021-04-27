"""
Module to test the managment of data used to train a model.
"""
import os
import unittest
import numpy as np
import torch
from brainspy.utils.pytorch import TorchUtils
from brainspy.utils.manager import (
    get_criterion,
    get_adam,
    get_optimizer,
    get_driver,
    get_algorithm,
)
from brainspy.processors.processor import Processor
from brainspy.algorithms.modules.optim import GeneticOptimizer
from brainspy.processors.hardware.drivers.cdaq import CDAQtoCDAQ


class ManagerTest(unittest.TestCase):

    """
    Tests for the manager.py class.

    """

    def __init__(self, test_name):
        super(ManagerTest, self).__init__()
        self.device = TorchUtils.get_device()

    def test_get_criterion(self):
        """
        Test for the get_criterion() method to return a valid fitness function.
        This method is tested with the test_torch.yaml file that defines a fitness criterion.
        This fitness function is tested with 2 arbitrary torch tensors namely "predictions" and "targets".
        """
        configs = {}
        configs["criterion"] = "corrsig_fit"
        criterion = get_criterion(configs)
        targets = torch.tensor(
            [
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [1.0],
                [1.0],
                [1.0],
            ],
            device=self.device,
        )
        predictions = torch.tensor(
            np.array(
                [
                    [-0.6571],
                    [-1.7871],
                    [-1.1129],
                    [-0.7235],
                    [0.3469],
                    [-0.5552],
                    [-0.3134],
                    [0.5727],
                    [1.4812],
                    [0.3388],
                    [0.9651],
                    [0.9615],
                ]
            ),
            device=self.device,
        )
        corsigfit = criterion(predictions, targets)
        assert isinstance(corsigfit, torch.Tensor)
        val = torch.tensor(12, device=self.device, dtype=torch.float64)
        self.assertEqual(corsigfit.shape, val.shape)

    def test_get_optimizer(self):
        """
        Test for the get_optimizer() method which returns an optimizer object based on the configs
        """

        NODE_CONFIGS = {}
        NODE_CONFIGS["processor_type"] = "simulation"
        NODE_CONFIGS["input_indices"] = [2, 3]
        NODE_CONFIGS["electrode_effects"] = {}
        NODE_CONFIGS["electrode_effects"]["amplification"] = 3
        NODE_CONFIGS["electrode_effects"]["clipping_value"] = [-300, 300]
        NODE_CONFIGS["electrode_effects"]["noise"] = {}
        NODE_CONFIGS["electrode_effects"]["noise"]["noise_type"] = "gaussian"
        NODE_CONFIGS["electrode_effects"]["noise"]["variance"] = 0.6533523201942444
        NODE_CONFIGS["driver"] = {}
        NODE_CONFIGS["waveform"] = {}
        NODE_CONFIGS["waveform"]["plateau_length"] = 1
        NODE_CONFIGS["waveform"]["slope_length"] = 0
        NODE_CONFIGS["optimizer"] = "genetic"
        NODE_CONFIGS["epochs"] = 100
        NODE_CONFIGS["partition"] = [4, 22]
        while "brains-py" not in os.getcwd():
            os.chdir("..")
            os.chdir("brains-py")
        model_dir = os.path.join(
            os.getcwd(), "tests/unit/utils/testfiles/training_data.pt"
        )
        model_data = torch.load(model_dir, map_location=torch.device("cpu"))
        model = Processor(
            NODE_CONFIGS,
            model_data["info"],
            model_data["model_state_dict"],
        )
        optim = get_optimizer(model, NODE_CONFIGS)
        assert isinstance(optim, GeneticOptimizer)

    def test_get_adam(self):
        """
        Test for the get_adam() method which returns an Adam optimizer object based on the configs
        """
        NODE_CONFIGS = {}
        NODE_CONFIGS["processor_type"] = "simulation"
        NODE_CONFIGS["input_indices"] = [2, 3]
        NODE_CONFIGS["electrode_effects"] = {}
        NODE_CONFIGS["electrode_effects"]["amplification"] = 3
        NODE_CONFIGS["electrode_effects"]["clipping_value"] = [-300, 300]
        NODE_CONFIGS["electrode_effects"]["noise"] = {}
        NODE_CONFIGS["electrode_effects"]["noise"]["noise_type"] = "gaussian"
        NODE_CONFIGS["electrode_effects"]["noise"]["variance"] = 0.6533523201942444
        NODE_CONFIGS["driver"] = {}
        NODE_CONFIGS["waveform"] = {}
        NODE_CONFIGS["waveform"]["plateau_length"] = 1
        NODE_CONFIGS["waveform"]["slope_length"] = 0
        NODE_CONFIGS["optimizer"] = "adam"
        NODE_CONFIGS["learning_rate"] = 0.001
        while "brains-py" not in os.getcwd():
            os.chdir("..")
            os.chdir("brains-py")
        model_dir = os.path.join(
            os.getcwd(), "tests/unit/utils/testfiles/training_data.pt"
        )
        model_data = torch.load(model_dir, map_location=torch.device("cpu"))
        model = Processor(
            NODE_CONFIGS,
            model_data["info"],
            model_data["model_state_dict"],
        )
        optim = get_adam(model, NODE_CONFIGS)
        assert isinstance(optim, torch.optim.Adam)

    def test_get_algorithm(self):
        """
        Test for the get_algorithm() method which returns a train function based on the configs. We test if this train function is callable

        """
        configs = {}
        configs["type"] = "genetic"
        algorithm = get_algorithm(configs)
        self.assertTrue(callable(algorithm))

    def test_get_driver(self):
        """
        Test for the get_driver() method which returns a driver object based on the configurations dictionary
        """
        configs = {}
        configs["instrument_type"] = "wrong_input"
        with self.assertRaises(
            NotImplementedError
        ):  # Testing for driver that does not exist
            get_driver(configs)

        # Testing for the driver type - CDAQtoCDAQ ( Test only when connected to Hardware )

        NODE_CONFIGS = {}
        NODE_CONFIGS["instrument_type"] = "cdaq_to_cdaq"
        NODE_CONFIGS["real_time_rack"] = False
        NODE_CONFIGS["sampling_frequency"] = 1000
        NODE_CONFIGS["instruments_setup"] = {}
        NODE_CONFIGS["instruments_setup"]["multiple_devices"] = False
        NODE_CONFIGS["instruments_setup"]["trigger_source"] = "cDAQ1/segment1"
        NODE_CONFIGS["instruments_setup"]["activation_instrument"] = "cDAQ1Mod3"
        NODE_CONFIGS["instruments_setup"]["activation_channels"] = [
            0,
            2,
            5,
            3,
            4,
            6,
            1,
        ]  # Channels through which voltages will be sent for activating the device (with both data inputs and control voltages)
        NODE_CONFIGS["instruments_setup"]["activation_voltages"] = [
            [-1.2, 0.6],
            [-1.2, 0.6],
            [-1.2, 0.6],
            [-1.2, 0.6],
            [-1.2, 0.6],
            [-0.7, 0.3],
            [-0.7, 0.3],
        ]
        NODE_CONFIGS["instruments_setup"]["readout_instrument"] = "cDAQ1Mod4"
        NODE_CONFIGS["instruments_setup"]["readout_channels"] = [4]
        # driver = get_driver(NODE_CONFIGS)
        # assert isinstance(driver, CDAQtoCDAQ)

    def runTest(self):
        self.test_get_criterion()
        self.test_get_optimizer()
        self.test_get_adam()
        self.test_get_algorithm()
        self.test_get_driver()


if __name__ == "__main__":
    unittest.main()
