"""
Module to test the managment of data used to train a model.
"""
import unittest
import numpy as np
import torch
import brainspy
from brainspy.utils.manager import (
    get_criterion,
    get_adam,
    get_optimizer,
    get_driver,
    get_algorithm,
)
from brainspy.processors.dnpu import DNPU
from brainspy.processors.processor import Processor
from brainspy.algorithms.ga import train as train_ga
from brainspy.algorithms.gd import train as train_gd
import brainspy.algorithms.modules.signal as criterions
from brainspy.algorithms.modules.optim import GeneticOptimizer
from brainspy.processors.hardware.drivers.cdaq import CDAQtoCDAQ
from brainspy.processors.hardware.drivers.nidaq import CDAQtoNiDAQ


class ManagerTest(unittest.TestCase):
    """
    Tests for the manager.py class. The class is is used to test methods
    required to acquire differnt functions,drivers,algorithms and optimizers.
    """
    """
    Tests for the get_criterion() method to return a valid fitness function.
    """
    def test_get_criterion_corrsig_fit(self):

        criterion = get_criterion("corrsig_fit")
        self.assertEqual(criterion, criterions.corrsig_fit)

    def test_criterion_accuracy_fit(self):

        criterion = get_criterion("accuracy_fit")
        self.assertEqual(criterion, criterions.accuracy_fit)

    def test_criterion_corr_fit(self):

        criterion = get_criterion("corr_fit")
        self.assertEqual(criterion, criterions.corr_fit)

    def test_criterion_corrsig(self):

        criterion = get_criterion("corrsig")
        self.assertEqual(criterion, criterions.corrsig)

    def test_criterion_fisher_fit(self):

        criterion = get_criterion("fisher_fit")
        self.assertEqual(criterion, criterions.fisher_fit)

    def test_criterion_fisher(self):

        criterion = get_criterion("fisher")
        self.assertEqual(criterion, criterions.fisher)

    def test_criterion_sigmoid_nn_distance(self):

        criterion = get_criterion("sigmoid_nn_distance")
        self.assertEqual(criterion, criterions.sigmoid_nn_distance)

    def test_criterion_bce(self):

        criterion = get_criterion("bce")
        assert isinstance(criterion, torch.nn.BCEWithLogitsLoss)

    def test_get_criterion_None(self):
        """
        NotImplementedError is raised if a none type is provided for a fitness function
        """
        configs = {}
        configs["criterion"] = None
        with self.assertRaises(NotImplementedError):
            get_criterion(configs)

    def test_get_criterion_wrong_input(self):
        """
        NotImplementedError is raised if the wrong fitness function name is provided
        """
        configs = {}
        configs["criterion"] = "wrong_input"
        with self.assertRaises(NotImplementedError):
            get_criterion(configs)

    def test_get_optimizer_from_custom_model(self):
        """
        Test for the get_optimizer() method which returns a genetic optimizer for a custom model
        """
        configs = self.get_default_node_configs()
        info = self.get_default_info_dict()
        node = Processor(configs, info)
        custom_model = DNPU(node, data_input_indices=[[3, 4]])
        configs["optimizer"] = "genetic"
        configs["partition"] = [4, 22]
        configs["epochs"] = 1000
        optimizer = get_optimizer(custom_model, configs)
        assert isinstance(optimizer, GeneticOptimizer)

    def test_get_genetic_optimizer_wrong_model(self):
        """
        AttributeError is raised if a genetic optimizer is called with the incorrect model
        A DNPU instance has to be provided if the configs contain only partition and epochs
        """
        model = torch.nn.Linear(20, 40)
        configs = {}
        configs["partition"] = [4, 22]
        configs["epochs"] = 1000
        configs["optimizer"] = "genetic"
        with self.assertRaises(AttributeError):
            get_optimizer(model, configs)

    def test_get_genetic_optimizer_with_configs(self):
        """
        Test to acquire a genetic optimizer by providing the partiion, epochs and gene range
        in the configs
        """
        model = torch.nn.Linear(20, 40)
        configs = {}
        configs["optimizer"] = "genetic"
        configs["partition"] = [4, 22]
        configs["epochs"] = 1000
        configs["gene_range"] = [
            [-0.55, 0.325],
            [-0.95, 0.55],
            [-1.0, 0.6],
            [-1.0, 0.6],
            [-1.0, 0.6],
            [-0.95, 0.55],
            [-0.55, 0.325],
        ]
        optimizer = get_optimizer(model, configs)
        assert isinstance(optimizer, GeneticOptimizer)

    def test_get_genetic_optimizer_wrong_gene_range(self):
        """
        Test to acquire a genetic optimizer by providing an wrong shape for the gene range
        """
        model = torch.nn.Linear(20, 40)
        configs = {}
        configs["optimizer"] = "genetic"
        configs["partition"] = [10, 22, 30]
        configs["epochs"] = 1000
        configs["gene_range"] = [0.5, 1.3, 1.3, 0.4, 0.5, 0.5]
        with self.assertRaises(IndexError):
            get_optimizer(model, configs)

    def test_get_adam(self):
        """
        Test for the get_adam() method which returns an Adam optimizer
        """
        configs = {}
        configs["optimizer"] = "adam"
        configs["learning_rate"] = 0.001
        optim = get_adam(torch.nn.Linear(20, 44), configs)
        assert isinstance(optim, torch.optim.Adam)

    def test_get_optimizer_missing_model(self):
        """
        AttributeError is raised if the model is not provided
        """
        configs = {}
        configs["optimizer"] = "adam"
        configs["learning_rate"] = 0.001
        with self.assertRaises(AttributeError):
            get_adam(None, configs)

        configs = {}
        configs["optimizer"] = "genetic"
        configs["partition"] = [10, 22, 30]
        configs["epochs"] = 1000
        with self.assertRaises(AttributeError):
            get_adam(None, configs)

    def test_get_optimizer_none(self):
        """
        AssertionError is raised if the optimizer name is not correct or is none
        """
        configs = {}
        configs["optimizer"] = None
        with self.assertRaises(AssertionError):
            get_optimizer(torch.nn.Linear(20, 44), configs)

        configs = {}
        configs["optimizer"] = "wrong_input"
        with self.assertRaises(AssertionError):
            get_optimizer(torch.nn.Linear(20, 44), configs)

    def test_get_algorithm_ga(self):
        """
        Test for the get_algorithm() method which returns a genetic algorithm train function based
        on the configs.
        """

        algorithm = get_algorithm("genetic")
        self.assertEqual(algorithm, train_ga)
        self.assertTrue(callable(algorithm))

    def test_get_algorithm_gd(self):
        """
        Test for the get_algorithm() method which returns a gradient descent train function based
        on the configs.
        """

        algorithm = get_algorithm("gradient")
        self.assertEqual(algorithm, train_gd)
        self.assertTrue(callable(algorithm))

    def test_get_algorithm_fail(self):
        """
        Assertion error is raised if the algorithm is unknown or None
        """
        configs = {}
        configs["type"] = None
        with self.assertRaises(NotImplementedError):
            get_algorithm(configs)

        configs = {}
        configs["type"] = "wrong_input"
        with self.assertRaises(NotImplementedError):
            get_algorithm(configs)

    """
    Test for the get_driver() method which returns a driver object based on the configurations
    dictionary
    """

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
        configs["instruments_setup"] = {}
        configs["instruments_setup"]["multiple_devices"] = False
        configs["instruments_setup"]["trigger_source"] = "cDAQ1/segment1"
        configs["instruments_setup"]["activation_instrument"] = "cDAQ1Mod3"
        configs["driver"]["instruments_setup"][
            "activation_sampling_frequency"] = 1000
        configs["instruments_setup"]["activation_channels"] = [
            0,
            2,
            5,
            3,
            4,
            6,
            1,
        ]
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
        configs["driver"]["instruments_setup"][
            "readout_sampling_frequency"] = 1000
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
        configs["driver"]["instruments_setup"] = {}
        configs["driver"]["instruments_setup"]["multiple_devices"] = False
        configs["driver"]["instruments_setup"][
            "trigger_source"] = "cDAQ1/segment1"
        configs["driver"]["instruments_setup"][
            "activation_instrument"] = "cDAQ1Mod3"
        configs["driver"]["instruments_setup"][
            "activation_sampling_frequency"] = 1000
        configs["driver"]["instruments_setup"]["activation_channels"] = [
            0,
            2,
            5,
            3,
            4,
            6,
            1,
        ]
        configs["driver"]["instruments_setup"]["activation_voltages"] = [
            [-1.2, 0.6],
            [-1.2, 0.6],
            [-1.2, 0.6],
            [-1.2, 0.6],
            [-1.2, 0.6],
            [-0.7, 0.3],
            [-0.7, 0.3],
        ]
        configs["driver"]["instruments_setup"][
            "readout_instrument"] = "cDAQ1Mod4"
        configs["driver"]["instruments_setup"][
            "readout_sampling_frequency"] = 1000
        configs["driver"]["instruments_setup"]["readout_channels"] = [
            4
        ]  # Channels for reading the output current values
        configs["waveform"] = {}
        configs["waveform"]["plateau_length"] = 10
        configs["waveform"]["slope_length"] = 30
        driver = get_driver(configs)
        assert isinstance(driver, CDAQtoNiDAQ)

    def test_get_driver_fail(self):
        """
        NotImplementedError is raised if the wrong driver name is provided or is None
        """
        configs = {}
        configs["instrument_type"] = "wrong_input"
        with self.assertRaises(
                NotImplementedError):  # Testing for driver that does not exist
            get_driver(configs)

        configs = {}
        configs["instrument_type"] = None
        with self.assertRaises(
                NotImplementedError):  # Testing for driver that does not exist
            get_driver(configs)

    def test_driver_wrong_configs(self):
        """
        Key error is raised if the configs are not provided for the specified driver
        """
        configs = {}
        configs["processor_type"] = "cdaq_to_nidaq"
        with self.assertRaises(KeyError):
            get_driver(configs)

    def get_default_info_dict(self):
        """
        Helper function used for testing
        """
        info = {}
        info['model_structure'] = {}
        info['model_structure']['hidden_sizes'] = [90, 90, 90, 90, 90]
        info['model_structure']['D_in'] = 7
        info['model_structure']['D_out'] = 1
        info['model_structure']['activation'] = 'relu'
        info['electrode_info'] = {}
        info['electrode_info']['electrode_no'] = 8
        info['electrode_info']['activation_electrodes'] = {}
        info['electrode_info']['activation_electrodes']['electrode_no'] = 7
        info['electrode_info']['activation_electrodes'][
            'voltage_ranges'] = np.array([[-0.55, 0.325], [-0.95, 0.55],
                                          [-1., 0.6], [-1., 0.6], [-1., 0.6],
                                          [-0.95, 0.55], [-0.55, 0.325]])
        info['electrode_info']['output_electrodes'] = {}
        info['electrode_info']['output_electrodes']['electrode_no'] = 1
        info['electrode_info']['output_electrodes']['amplification'] = [28.5]
        info['electrode_info']['output_electrodes']['clipping_value'] = None

        return info

    def get_default_node_configs(self):
        """
        Helper function used for testing
        """
        configs = {}
        configs["processor_type"] = "simulation"
        configs["electrode_effects"] = {}
        configs["electrode_effects"]["clipping_value"] = None
        configs["driver"] = {}
        configs["waveform"] = {}
        configs["waveform"]["plateau_length"] = 1
        configs["waveform"]["slope_length"] = 0
        return configs


if __name__ == "__main__":
    unittest.main()
