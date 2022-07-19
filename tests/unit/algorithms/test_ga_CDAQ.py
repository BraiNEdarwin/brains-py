import unittest
import warnings
import torch
import brainspy
from brainspy.algorithms.ga import train, evaluate_population
from brainspy.processors.dnpu import DNPU
from brainspy.processors.hardware.processor import HardwareProcessor
from brainspy.utils.pytorch import TorchUtils
from brainspy.utils.performance.accuracy import zscore_norm
from brainspy.utils.performance.data import get_data
from brainspy.algorithms.modules.optim import GeneticOptimizer
from brainspy.processors.hardware.drivers.ni.setup import NationalInstrumentsSetup
from tests.unit.testing_utils import check_test_configs


class GA_Test_CDAQ(unittest.TestCase):
    """
    Tests for the Genetic Algorithm with a DNPU using a CDAQ driver

    To run this file, the device has to be connected to a CDAQ setup and
    the device configurations have to be specified depending on the setup.

    The test mode has to be set to HARDWARE_CDAQ in tests/main.py.
    The required keys have to be defined in the get_configs_CDAQ() function.

    Some sample keys have been defined to run tests which do not require connection
    to the hardware.
    """
    def get_configs_CDAQ(self):
        """
        Generate configurations to initialize the CDAQ driver.
        For this test file, The CDAQ driver should contain 7 inputs (activation instruments)
        and 1 output (readout instrument)
        """
        configs = {}
        configs["waveform"] = {}
        configs["waveform"]["plateau_length"] = 10
        configs["waveform"]["slope_length"] = 30

        configs["amplification"] = 100
        configs["inverted_output"] = True
        configs["output_clipping_range"] = [-1, 1]

        configs["processor_type"] = "cdaq_to_cdaq"
        configs["instrument_type"] = "cdaq_to_cdaq"

        configs["instruments_setup"] = {}

        configs["instruments_setup"]["multiple_devices"] = False
        # TODO Specify the name of the Trigger Source
        configs["instruments_setup"]["trigger_source"] = "a"

        # TODO Specify the name of the Activation instrument
        configs["instruments_setup"]["activation_instrument"] = "b"

        # TODO Specify the Activation channels (pin numbers)
        # For example, [1,2,3,4,5,6,7]
        configs["instruments_setup"]["activation_channels"] = [
            1, 2, 3, 4, 5, 6, 7
        ]

        # TODO Specify the activation Voltage ranges
        # For example, [[-1.2, 0.6],[-1.2, 0.6],[-1.2, 0.6],[-1.2, 0.6],[-1.2, 0.6],[-0.7, 0.3],[-0.7, 0.3]]
        configs["instruments_setup"]["activation_voltage_ranges"] = [
            [-1.2, 0.6], [-1.2, 0.6], [-1.2, 0.6], [-1.2, 0.6], [-1.2, 0.6],
            [-0.7, 0.3], [-0.7, 0.3]
        ]

        # TODO Specify the name of the Readout Instrument
        configs["instruments_setup"]["readout_instrument"] = "c"

        # TODO Specify the readout channels
        # For example, [4]
        configs["instruments_setup"]["readout_channels"] = [4]
        configs["instruments_setup"]["activation_sampling_frequency"] = 500
        configs["instruments_setup"]["readout_sampling_frequency"] = 1000
        configs["instruments_setup"]["average_io_point_difference"] = True

        return configs

    def get_train_parameters(self):
        """
        Generate some random train parameters for Genetic algorithm
        """
        results = {}
        threshhold = 1000
        size = torch.randint(1, threshhold, (1, 1)).item()
        results["inputs"] = TorchUtils.format(torch.rand((size, 2)))
        results["targets"] = TorchUtils.format(torch.randint(0, 2, (size, 1)))
        results["norm_inputs"] = zscore_norm(results["inputs"])
        dataloader = get_data(results, batch_size=512)
        dataloaders = [dataloader]
        criterion = torch.nn.MSELoss()
        optimizer = GeneticOptimizer(gene_ranges=TorchUtils.format(
            torch.tensor([[-1.2, 0.6], [-1.2, 0.6], [-1.2, 0.6], [-1.2, 0.6],
                          [-1.2, 0.6]])),
                                     partition=[4, 22],
                                     epochs=100)
        configs = {}
        configs["epochs"] = 100
        configs["stop_threshold"] = 0.5
        return dataloaders, criterion, optimizer, configs

    @unittest.skipUnless(brainspy.TEST_MODE == "HARDWARE_CDAQ",
                         "Hardware test is skipped for simulation setup.")
    def test_train_cdaq(self):
        """
        Test for genetic algorithm with random inputs using a CDAQ model
        """
        configs = self.get_configs_CDAQ()
        hp = HardwareProcessor(
            configs,
            slope_length=configs["waveform"]["slope_length"],
            plateau_length=configs["waveform"]["plateau_length"],
        )
        model = DNPU(hp, [[1, 2], [1, 3], [3, 4]])
        dataloaders, criterion, optimizer, configs = self.get_train_parameters(
        )
        try:
            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always")
                model, results = train(model, dataloaders, criterion,
                                       optimizer, configs)
                self.assertEqual(len(caught_warnings), 1)
        except (Exception):
            self.fail("Could not run Genetic Algorithm")
        else:
            self.assertTrue("best_result_index" in results)
            self.assertTrue("genome_history" in results)
            self.assertTrue("performance_history" in results)
            self.assertTrue("correlation_history" in results)
            self.assertTrue("best_output" in results)

    def test_train_invalid_model(self):
        """
        Running genetic algorithm on an external pytorch model raises an Exception
        """
        with self.assertRaises(Exception):
            model = torch.nn.Linear((1, 1))
            dataloaders, criterion, optimizer, configs = self.get_train_parameters(
            )
            train(model, dataloaders, criterion, optimizer, configs)

    def test_train_invalid_model_type(self):
        """
        Wrong type of model raises an AssertionError
        """
        models = [{}, [1, 2, 3], "invalid type", 100, 5.5]
        dataloaders, criterion, optimizer, configs = self.get_train_parameters(
        )
        for model in models:
            with self.assertRaises(AssertionError):
                train(model, dataloaders, criterion, optimizer, configs)

    @unittest.skipUnless(brainspy.TEST_MODE == "HARDWARE_CDAQ",
                         "Hardware test is skipped for simulation setup.")
    def test_train_invalid_dataloader_type(self):
        """
        Wrong type of dataloader raises an AssertionError
        """
        configs = self.get_configs_CDAQ()
        hp = HardwareProcessor(
            configs,
            slope_length=configs["waveform"]["slope_length"],
            plateau_length=configs["waveform"]["plateau_length"],
        )
        model = DNPU(hp, [[1, 2], [1, 3], [3, 4]])
        dataloader, criterion, optimizer, configs = self.get_train_parameters()
        dataloaders = [dataloader[0], {}, 100, "invalid type"]
        for d in dataloaders:
            with self.assertRaises(AssertionError):
                train(model, d, criterion, optimizer, configs)

    @unittest.skipUnless(brainspy.TEST_MODE == "HARDWARE_CDAQ",
                         "Hardware test is skipped for simulation setup.")
    def test_train_invalid_optimizer_type(self):
        """
        Wrong type of optimizer raises an AssertionError
        """
        configs = self.get_configs_CDAQ()
        hp = HardwareProcessor(
            configs,
            slope_length=configs["waveform"]["slope_length"],
            plateau_length=configs["waveform"]["plateau_length"],
        )
        model = DNPU(hp, [[1, 2], [1, 3], [3, 4]])
        dataloaders, criterion, optimizer, configs = self.get_train_parameters(
        )
        optimizers = ["invalid type", 100, [1, 2, 3, 4, 5], {}]
        for o in optimizers:
            with self.assertRaises(AssertionError):
                train(model, dataloaders, criterion, o, configs)

    @unittest.skipUnless(brainspy.TEST_MODE == "HARDWARE_CDAQ",
                         "Hardware test is skipped for simulation setup.")
    def test_train_invalid_criterion_type(self):
        """
        Wrong type of criterion raises an AssertionError
        """
        configs = self.get_configs_CDAQ()
        hp = HardwareProcessor(
            configs,
            slope_length=configs["waveform"]["slope_length"],
            plateau_length=configs["waveform"]["plateau_length"],
        )
        model = DNPU(hp, [[1, 2], [1, 3], [3, 4]])
        dataloaders, criterion, optimizer, configs = self.get_train_parameters(
        )
        criterions = ["invalid type", 100, [1, 2, 3, 4, 5], {}]
        for c in criterions:
            with self.assertRaises(AssertionError):
                train(model, dataloaders, c, optimizer, configs)

    @unittest.skipUnless(brainspy.TEST_MODE == "HARDWARE_CDAQ",
                         "Hardware test is skipped for simulation setup.")
    def test_train_invalid_configs_type(self):
        """
        Wrong type of configs raises an AssertionError
        """
        configs = self.get_configs_CDAQ()
        hp = HardwareProcessor(
            configs,
            slope_length=configs["waveform"]["slope_length"],
            plateau_length=configs["waveform"]["plateau_length"],
        )
        model = DNPU(hp, [[1, 2], [1, 3], [3, 4]])
        dataloaders, criterion, optimizer, configs = self.get_train_parameters(
        )
        configs = ["invalid type", 100, [1, 2, 3, 4, 5]]
        for config in configs:
            with self.assertRaises(AssertionError):
                train(model, dataloaders, criterion, optimizer, config)

    @unittest.skipUnless(brainspy.TEST_MODE == "HARDWARE_CDAQ",
                         "Hardware test is skipped for simulation setup.")
    def test_evaluate_population_CDAQ(self):
        """
        Test for the evaluate_population method
        """
        try:
            configs = self.get_configs_CDAQ()
            hp = HardwareProcessor(
                configs,
                slope_length=configs["waveform"]["slope_length"],
                plateau_length=configs["waveform"]["plateau_length"],
            )
            model = DNPU(hp, [[1, 2], [1, 3], [3, 4]])
            dataloaders, criterion, optimizer, configs = self.get_train_parameters(
            )
            inputs, targets = dataloaders[0].dataset[:]
            inputs, targets = TorchUtils.format(inputs), TorchUtils.format(
                targets)
            outputs, criterion_pool = evaluate_population(
                inputs, targets, optimizer.pool, model, criterion)
        except (Exception):
            self.fail("Could not execute function for the CDAQ model")
        else:
            self.assertTrue(outputs is not None)
            self.assertIsInstance(outputs, torch.Tensor)
            self.assertTrue(criterion_pool is not None)
            self.assertIsInstance(criterion_pool, torch.Tensor)

    @unittest.skipUnless(brainspy.TEST_MODE == "HARDWARE_CDAQ",
                         "Hardware test is skipped for simulation setup.")
    def test_evaluate_population_fail(self):
        """
        Invalid type for inputs or targets raises an AssertionError
        """
        inputs = ["invalid type", 100, [1, 2, 3, 4, 5], {}]
        targets = ["invalid type", 100, [1, 2, 3, 4, 5], {}]
        dataloaders, criterion, optimizer, configs = self.get_train_parameters(
        )
        input, target = dataloaders[0].dataset[:]
        input, target = TorchUtils.format(input), TorchUtils.format(target)
        configs = self.get_configs_CDAQ()
        hp = HardwareProcessor(
            configs,
            slope_length=configs["waveform"]["slope_length"],
            plateau_length=configs["waveform"]["plateau_length"],
        )
        model = DNPU(hp, [[1, 2], [1, 3], [3, 4]])
        dataloaders, criterion, optimizer, configs = self.get_train_parameters(
        )
        for i in inputs:
            with self.assertRaises(AssertionError):
                evaluate_population(i, target, optimizer.pool, model,
                                    criterion)
        for t in targets:
            with self.assertRaises(AssertionError):
                evaluate_population(input, t, optimizer.pool, model, criterion)


if __name__ == "__main__":

    testobj = GA_Test_CDAQ()
    configs = testobj.get_configs_CDAQ()
    try:
        NationalInstrumentsSetup.type_check(configs)
        if check_test_configs(configs):
            raise unittest.SkipTest("Configs are missing. Skipping all tests.")
        else:
            unittest.main()
    except (Exception):
        print(Exception)
        raise unittest.SkipTest(
            "Configs not specified correctly. Skipping all tests.")
