"""
Module to test the managment of data used to train a model.
"""
import os
import unittest
import numpy as np
import torch
import brainspy
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
from brainspy.processors.hardware.processor import HardwareProcessor
from brainspy.processors.simulation.processor import SurrogateModel
from brainspy.processors.hardware.drivers.cdaq import CDAQtoCDAQ
from brainspy.processors.hardware.drivers.nidaq import CDAQtoNiDAQ


class ManagerTest(unittest.TestCase):

    """
    Tests for the manager.py class.

    """

    def __init__(self, test_name):
        super(ManagerTest, self).__init__()
        self.device = TorchUtils.get_device()

    """
    Tests for the get_criterion() method to return a valid fitness function.
    """

    def test_get_criterion_corrsig_fit(self):

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

    def test_criterion_accuracy_fit(self):
        configs = {}
        configs["criterion"] = "accuracy_fit"
        criterion = get_criterion(configs)
        accuracy_fit = criterion(torch.rand((100, 1)), torch.rand(100, 1))
        assert isinstance(accuracy_fit, torch.Tensor)
        self.assertEqual(accuracy_fit.shape, torch.Size([]))

        accuracy_fit = criterion(torch.rand((100, 1)), torch.rand(100, 1), True)
        assert isinstance(accuracy_fit, torch.Tensor)
        val = torch.tensor(0, device=self.device, dtype=torch.float64)
        self.assertEqual(accuracy_fit, val)

    def test_criterion_corr_fit(self):
        configs = {}
        configs["criterion"] = "corr_fit"
        criterion = get_criterion(configs)
        corr_fit = criterion(torch.rand((100, 1)), torch.rand(100, 1))
        assert isinstance(corr_fit, torch.Tensor)
        self.assertEqual(corr_fit.shape, torch.Size([]))

        corr_fit = criterion(torch.rand((100, 1)), torch.rand(100, 1), True)
        assert isinstance(corr_fit, torch.Tensor)
        val = torch.tensor(-1, device=self.device, dtype=torch.float64)
        self.assertEqual(corr_fit, val)

    def test_criterion_corrsig(self):
        configs = {}
        configs["criterion"] = "corrsig"
        criterion = get_criterion(configs)
        corrsig = criterion(torch.rand((100, 1)), torch.round(torch.rand((100, 1))))
        assert isinstance(corrsig, torch.Tensor)
        self.assertEqual(corrsig.shape, torch.Size([]))

    def test_criterion_fisher_fit(self):
        configs = {}
        configs["criterion"] = "fisher_fit"
        criterion = get_criterion(configs)
        fisher_fit = criterion(torch.rand((100, 1)), torch.rand((100, 1)), False)
        assert isinstance(fisher_fit, torch.Tensor)
        self.assertEqual(fisher_fit.shape, torch.Size([]))

        fisher_fit = criterion(torch.rand((100, 1)), torch.rand((100, 1)), True)
        assert isinstance(fisher_fit, torch.Tensor)
        val = torch.tensor(0, device=self.device, dtype=torch.float64)
        self.assertEqual(fisher_fit, val)

    def test_criterion_fisher(self):
        configs = {}
        configs["criterion"] = "fisher"
        criterion = get_criterion(configs)
        fisher = criterion(torch.rand((100, 1)), torch.round(torch.rand((100, 1))))
        assert isinstance(fisher, torch.Tensor)
        self.assertEqual(fisher.shape, torch.Size([]))

    def test_criterion_sigmoid_nn_distance(self):
        configs = {}
        configs["criterion"] = "sigmoid_nn_distance"
        criterion = get_criterion(configs)
        sigmoid_nn_distance = criterion(torch.rand((100, 1)))
        assert isinstance(sigmoid_nn_distance, torch.Tensor)
        self.assertEqual(sigmoid_nn_distance.shape, torch.Size([]))

    def test_criterion_bce(self):
        configs = {}
        configs["criterion"] = "bce"
        criterion = get_criterion(configs)
        target = torch.ones([10, 64], dtype=torch.float32)
        output = torch.full([10, 64], 1.5)
        bce = criterion(target, output)
        assert isinstance(bce, torch.Tensor)
        self.assertEqual(bce.shape, torch.Size([]))

    def test_get_optimizer(self):
        """
        Test for the get_optimizer() method which returns an optimizer object based on the configs
        """
        configs = {}
        configs["processor_type"] = "simulation_debug"
        configs["input_indices"] = [2, 3]
        configs["electrode_effects"] = {}
        configs["electrode_effects"]["amplification"] = 28.5
        configs["electrode_effects"]["clipping_value"] = [-110, 110]
        configs["electrode_effects"]["noise"] = {}
        configs["electrode_effects"]["noise"]["noise_type"] = "gaussian"
        configs["electrode_effects"]["noise"]["variance"] = 0.6533523201942444
        configs["waveform"] = {}
        configs["waveform"]["plateau_length"] = 10
        configs["waveform"]["slope_length"] = 30
        configs["optimizer"] = "genetic"
        configs["epochs"] = 100
        configs["partition"] = [4, 22]
        self.configs = configs
        self.model_data = {}
        self.model_data["info"] = {}
        self.model_data["info"]["model_structure"] = {
            "hidden_sizes": [90, 90, 90],
            "D_in": 7,
            "D_out": 1,
            "activation": "relu",
        }

        self.debug_model = SurrogateModel(self.model_data["info"]["model_structure"])
        model = HardwareProcessor(
            self.debug_model,
            slope_length=self.configs["waveform"]["slope_length"],
            plateau_length=self.configs["waveform"]["plateau_length"],
        )
        self.model = TorchUtils.format(model)
        optim = get_optimizer(model, configs)
        assert isinstance(optim, GeneticOptimizer)

    def test_get_adam(self):
        """
        Test for the get_adam() method which returns an Adam optimizer object based on the configs
        """
        configs = {}
        configs["processor_type"] = "simulation_debug"
        configs["input_indices"] = [2, 3]
        configs["electrode_effects"] = {}
        configs["electrode_effects"]["amplification"] = 28.5
        configs["electrode_effects"]["clipping_value"] = [-110, 110]
        configs["electrode_effects"]["noise"] = {}
        configs["electrode_effects"]["noise"]["noise_type"] = "gaussian"
        configs["electrode_effects"]["noise"]["variance"] = 0.6533523201942444
        configs["waveform"] = {}
        configs["waveform"]["plateau_length"] = 10
        configs["waveform"]["slope_length"] = 30
        configs["optimizer"] = "adam"
        configs["learning_rate"] = 0.001

        self.configs = configs
        self.model_data = {}
        self.model_data["info"] = {}
        self.model_data["info"]["model_structure"] = {
            "hidden_sizes": [90, 90, 90],
            "D_in": 7,
            "D_out": 1,
            "activation": "relu",
        }

        self.debug_model = SurrogateModel(self.model_data["info"]["model_structure"])
        model = HardwareProcessor(
            self.debug_model,
            slope_length=self.configs["waveform"]["slope_length"],
            plateau_length=self.configs["waveform"]["plateau_length"],
        )
        self.model = TorchUtils.format(model)
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

        configs = {}
        configs["type"] = "gradient"
        algorithm = get_algorithm(configs)
        self.assertTrue(callable(algorithm))

    def test_get_driver_fail(self):
        """
        Test for the get_driver() method which returns a driver object based on the configurations dictionary
        """
        configs = {}
        configs["instrument_type"] = "wrong_input"
        with self.assertRaises(
            NotImplementedError
        ):  # Testing for driver that does not exist
            get_driver(configs)

    @unittest.skipUnless(
        brainspy.TEST_MODE == "HARDWARE_CDAQ",
        "Method deactivated as it is only possible to be tested on a CDAQ TO CDAQ setup",
    )
    def test_get_driver_cdaq_to_cdaq(self):
        """
        Testing for the driver type - CDAQtoCDAQ
        """
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
        configs["instruments_setup"]["activation_voltage_ranges"] = [
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
        driver = get_driver(configs)
        assert isinstance(driver, CDAQtoCDAQ)

    @unittest.skipUnless(
        brainspy.TEST_MODE == "HARDWARE_NIDAQ",
        "Method deactivated as it is only possible to be tested on a NIDAQ TO NIDAQ setup",
    )
    def test_get_driver_cdaq_to_nidaq(self):
        """
        Testing for the driver type - CDAQtoCDAQ
        """
        configs = {}
        configs["processor_type"] = "cdaq_to_nidaq"
        configs["input_indices"] = [2, 3]
        configs["electrode_effects"] = {}
        configs["electrode_effects"]["amplification"] = 3
        configs["electrode_effects"]["clipping_value"] = [-300, 300]
        configs["electrode_effects"]["noise"] = {}
        configs["electrode_effects"]["noise"]["noise_type"] = "gaussian"
        configs["electrode_effects"]["noise"]["variance"] = 0.6533523201942444
        configs["driver"] = {}
        configs["driver"]["real_time_rack"] = False
        configs["driver"]["sampling_frequency"] = 1000
        configs["driver"]["instruments_setup"] = {}
        configs["driver"]["instruments_setup"]["multiple_devices"] = False
        configs["driver"]["instruments_setup"]["trigger_source"] = "cDAQ1/segment1"
        configs["driver"]["instruments_setup"]["activation_instrument"] = "cDAQ1Mod3"
        configs["driver"]["instruments_setup"]["activation_channels"] = [
            0,
            2,
            5,
            3,
            4,
            6,
            1,
        ]  # Channels through which voltages will be sent for activating the device (with both data inputs and control voltages)
        configs["driver"]["instruments_setup"]["activation_voltages"] = [
            [-1.2, 0.6],
            [-1.2, 0.6],
            [-1.2, 0.6],
            [-1.2, 0.6],
            [-1.2, 0.6],
            [-0.7, 0.3],
            [-0.7, 0.3],
        ]
        configs["driver"]["instruments_setup"]["readout_instrument"] = "cDAQ1Mod4"
        configs["driver"]["instruments_setup"]["readout_channels"] = [
            4
        ]  # Channels for reading the output current values

        configs["waveform"] = {}
        configs["waveform"]["plateau_length"] = 10
        configs["waveform"]["slope_length"] = 30
        driver = get_driver(configs)
        assert isinstance(driver, CDAQtoNiDAQ)

    def runTest(self):
        self.test_get_criterion_corrsig_fit()
        self.test_criterion_accuracy_fit()
        self.test_criterion_corr_fit()
        self.test_criterion_corrsig()
        self.test_criterion_fisher_fit()
        self.test_criterion_fisher()
        self.test_criterion_bce()
        # self.test_get_optimizer()
        self.test_get_adam()
        self.test_get_algorithm()
        self.test_get_driver_fail()
        self.test_get_driver_cdaq_to_cdaq()
        self.test_get_driver_cdaq_to_nidaq()


if __name__ == "__main__":
    unittest.main()
