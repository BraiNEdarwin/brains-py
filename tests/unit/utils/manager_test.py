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
from brainspy.processors.dnpu import DNPU
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

        configs = {}
        configs["processor_type"] = "simulation"
        configs["input_indices"] = [[2, 3]]
        configs["electrode_effects"] = {}
        configs["electrode_effects"]["amplification"] = [28.5]
        configs["electrode_effects"]["clipping_value"] = [-110, 110]
        configs["electrode_effects"]["noise"] = {}
        configs["electrode_effects"]["noise"]["type"] = "gaussian"
        configs["electrode_effects"]["noise"]["variance"] = 0.6533523201942444
        configs["driver"] = {}
        configs["waveform"] = {}
        configs["waveform"]["plateau_length"] = 1
        configs["waveform"]["slope_length"] = 0
        configs["optimizer"] = "genetic"
        configs["epochs"] = 100
        configs["partition"] = [4, 22]
        while "brains-py" not in os.getcwd():
            os.chdir("..")
            os.chdir("brains-py")
        model_dir = os.path.join(
            os.getcwd(), "tests/unit/utils/testfiles/training_data.pt"
        )
        model_data = torch.load(model_dir, map_location=TorchUtils.get_device())
        processor = Processor(
            configs,
            model_data["info"],
            model_data["model_state_dict"],
        )
        model = DNPU(processor=processor, data_input_indices=configs['input_indices'])
        optim = get_optimizer(model, configs)
        assert isinstance(optim, GeneticOptimizer)

    def test_get_adam(self):
        """
        Test for the get_adam() method which returns an Adam optimizer object based on the configs
        """
        configs = {}
        configs["processor_type"] = "simulation"
        configs["input_indices"] = [[2, 3]]
        configs["electrode_effects"] = {}
        configs["electrode_effects"]["amplification"] = [28.5]
        configs["electrode_effects"]["clipping_value"] = [-110, 110]
        configs["electrode_effects"]["noise"] = {}
        configs["electrode_effects"]["noise"]["type"] = "gaussian"
        configs["electrode_effects"]["noise"]["variance"] = 0.6533523201942444
        configs["driver"] = {}
        configs["waveform"] = {}
        configs["waveform"]["plateau_length"] = 1
        configs["waveform"]["slope_length"] = 0
        configs["optimizer"] = "adam"
        configs["learning_rate"] = 0.001
        while "brains-py" not in os.getcwd():
            os.chdir("..")
            os.chdir("brains-py")
        model_dir = os.path.join(
            os.getcwd(), "tests/unit/utils/testfiles/training_data.pt"
        )
        model_data = torch.load(model_dir, map_location=TorchUtils.get_device())
        processor = Processor(
            configs,
            model_data["info"],
            model_data["model_state_dict"],
        )
        model = DNPU(processor=processor, data_input_indices=configs['input_indices'])
        optim = get_adam(model, configs)
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

        configs = {}
        configs["instrument_type"] = "cdaq_to_cdaq"
        configs["real_time_rack"] = False
        configs["sampling_frequency"] = 1000
        configs["instruments_setup"] = {}
        configs["instruments_setup"]["multiple_devices"] = False
        configs["instruments_setup"]["trigger_source"] = "cDAQ1/segment1"
        configs["instruments_setup"]["activation_instrument"] = "cDAQ1Mod3"
        configs["instruments_setup"]["activation_channels"] = [
            0,
            2,
            5,
            3,
            4,
            6,
            1,
        ]  # Channels through which voltages will be sent for activating the device (with both data inputs and control voltages)
        configs["instruments_setup"]["activation_voltages"] = [
            [-1.2, 0.6],
            [-1.2, 0.6],
            [-1.2, 0.6],
            [-1.2, 0.6],
            [-1.2, 0.6],
            [-0.7, 0.3],
            [-0.7, 0.3],
        ]
        configs["instruments_setup"]["readout_instrument"] = "cDAQ1Mod4"
        configs["instruments_setup"]["readout_channels"] = [4]
        # driver = get_driver(configs)
        # assert isinstance(driver, CDAQtoCDAQ)

    def runTest(self):
        self.test_get_criterion()
        self.test_get_optimizer()
        self.test_get_adam()
        self.test_get_algorithm()
        self.test_get_driver()


if __name__ == "__main__":
    unittest.main()
