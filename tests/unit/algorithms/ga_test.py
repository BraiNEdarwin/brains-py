import unittest
import torch
import brainspy.algorithms.modules.signal as function
from brainspy.algorithms.ga import train, evaluate_population
from brainspy.utils.pytorch import TorchUtils
from brainspy.algorithms.modules.performance.accuracy import zscore_norm
from brainspy.algorithms.modules.performance.data import get_data
from brainspy.algorithms.modules.optim import GeneticOptimizer
from brainspy.processors.hardware.drivers.nidaq import CDAQtoNiDAQ
from brainspy.processors.hardware.drivers.cdaq import CDAQtoCDAQ
from brainspy.processors.simulation.processor import SurrogateModel
from brainspy.processors.hardware.processor import HardwareProcessor


class GA_Test(unittest.TestCase):

    def get_train_parameters(self):
        """
        Generate t6he train parameters for Genetic algorithm
        """
        results = {}
        threshhold = 10000
        size = torch.randint(0, threshhold, (1, 1)).item()
        results["inputs"] = TorchUtils.format(torch.rand((size, 1)))
        results["targets"] = TorchUtils.format(torch.randint(0, 2, (size, 1)))
        results["norm_inputs"] = zscore_norm(results["inputs"])
        dataloader = get_data(results, batch_size=512)
        dataloaders = [dataloader]
        criterion = function.accuracy_fit
        optimizer = GeneticOptimizer(gene_ranges=TorchUtils.format(
            torch.tensor([[-1.2, 0.6], [-1.2, 0.6]])),
            partition=[4, 22],
            epochs=100)
        configs = {}
        configs["epochs"] = 100
        configs["stop_threshold"] = 0.5
        return dataloaders, criterion, optimizer, configs

    def get_configs_NIDAQ(self):
        """
        Generate configurations to initialize the NIDAQ driver
        Devices - .. (not tested yet)
        """
        configs = {}
        configs["instrument_type"] = "cdaq_to_nidaq"
        configs["real_time_rack"] = False
        configs["inverted_output"] = True
        configs["amplification"] = 100
        configs["instruments_setup"] = {}
        configs["instruments_setup"]["multiple_devices"] = False
        configs["instruments_setup"]["trigger_source"] = "cDAQ3/segment1"
        configs["instruments_setup"]["activation_instrument"] = "cDAQ3Mod1"
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
        configs["instruments_setup"]["readout_instrument"] = "cDAQ3Mod2"
        configs["instruments_setup"]["readout_channels"] = [4]
        configs["instruments_setup"]["activation_sampling_frequency"] = 500
        configs["instruments_setup"]["readout_sampling_frequency"] = 1000
        configs["instruments_setup"]["average_io_point_difference"] = True
        return configs

    def get_configs_CDAQ(self):
        """
        Generate configurations to initialize the CDAQ driver
        Devices used - cDAQ3Mod1,cDAQ3Mod2
        """
        configs = {}
        configs["instrument_type"] = "cdaq_to_cdaq"
        configs["real_time_rack"] = False
        configs["inverted_output"] = True
        configs["amplification"] = 100
        configs["output_clipping_range"] = [-1, 1]
        configs["instruments_setup"] = {}
        configs["instruments_setup"]["multiple_devices"] = False
        configs["instruments_setup"]["trigger_source"] = "cDAQ3/segment1"
        configs["instruments_setup"]["activation_instrument"] = "cDAQ3Mod1"
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
        configs["instruments_setup"]["readout_instrument"] = "cDAQ3Mod2"
        configs["instruments_setup"]["readout_channels"] = [4]
        configs["instruments_setup"]["activation_sampling_frequency"] = 500
        configs["instruments_setup"]["readout_sampling_frequency"] = 1000
        configs["instruments_setup"]["average_io_point_difference"] = True
        return configs

    def test_train_nidaq(self):
        """
        Test for genetic algorithm with random inputs using a NIDAQ model
        """
        model = CDAQtoNiDAQ(self.get_configs_NIDAQ())
        dataloaders, criterion, optimizer, configs = self.get_train_parameters(
        )
        try:
            model, results = train(model, dataloaders, criterion, optimizer,
                                   configs)
        except (Exception):
            self.fail("Could not run Genetic Algorithm")
        else:
            self.assertTrue("best_result_index" in results)
            self.assertTrue("genome_history" in results)
            self.assertTrue("performance_history" in results)
            self.assertTrue("correlation_history" in results)
            self.assertTrue("best_output" in results)

    def test_train_cdaq(self):
        """
        Test for genetic algorithm with random inputs using a CDAQ model
        """
        model = CDAQtoCDAQ(self.get_configs_CDAQ())
        dataloaders, criterion, optimizer, configs = self.get_train_parameters(
        )
        try:
            model, results = train(model, dataloaders, criterion, optimizer,
                                   configs)
        except (Exception):
            self.fail("Could not run Genetic Algorithm")
        else:
            self.assertTrue("best_result_index" in results)
            self.assertTrue("genome_history" in results)
            self.assertTrue("performance_history" in results)
            self.assertTrue("correlation_history" in results)
            self.assertTrue("best_output" in results)

    def test_train_surrogate_model_processor(self):

        configs = {}
        configs["waveform"] = {}
        configs["waveform"]["plateau_length"] = 10
        configs["waveform"]["slope_length"] = 30

        model_data = {}
        model_data["info"] = {}
        model_data["info"]["model_structure"] = {
            "hidden_sizes": [90, 90, 90],
            "D_in": 7,
            "D_out": 1,
            "activation": "relu",
        }
        surrogate_model = SurrogateModel(
            model_data["info"]["model_structure"])
        model = HardwareProcessor(
            surrogate_model,
            slope_length=configs["waveform"]["slope_length"],
            plateau_length=configs["waveform"]["plateau_length"],
        )
        dataloaders, criterion, optimizer, configs = self.get_train_parameters(
        )
        try:
            model, results = train(model, dataloaders, criterion, optimizer,
                                   configs)
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

    def test_train_invalid_dataloader_type(self):
        """
        Wrong type of dataloader raises an AssertionError
        """
        model = CDAQtoNiDAQ(self.get_configs_NIDAQ())
        dataloader, criterion, optimizer, configs = self.get_train_parameters()
        dataloaders = [dataloader[0], {}, [1, 2, 3, 4], 100, "invalid type"]
        for d in dataloaders:
            with self.assertRaises(AssertionError):
                train(model, d, criterion, optimizer, configs)

    def test_train_invalid_optimizer_type(self):
        """
        Wrong type of optimizer raises an AssertionError
        """
        model = CDAQtoNiDAQ(self.get_configs_NIDAQ())
        dataloaders, criterion, optimizer, configs = self.get_train_parameters(
        )
        optimizers = ["invalid type", 100, [1, 2, 3, 4, 5], {}]
        for o in optimizers:
            with self.assertRaises(AssertionError):
                train(model, dataloaders, criterion, o, configs)

    def test_train_invalid_criterion_type(self):
        """
        Wrong type of criterion raises an AssertionError
        """
        model = CDAQtoNiDAQ(self.get_configs_NIDAQ())
        dataloaders, criterion, optimizer, configs = self.get_train_parameters(
        )
        criterions = ["invalid type", 100, [1, 2, 3, 4, 5], {}]
        for c in criterions:
            with self.assertRaises(AssertionError):
                train(model, dataloaders, c, optimizer, configs)

    def test_train_invalid_configs_type(self):
        """
        Wrong type of configs raises an AssertionError
        """

        model = CDAQtoNiDAQ(self.get_configs_NIDAQ())
        dataloaders, criterion, optimizer, configs = self.get_train_parameters(
        )
        configs = ["invalid type", 100, [1, 2, 3, 4, 5]]
        for config in configs:
            with self.assertRaises(AssertionError):
                train(model, dataloaders, criterion, optimizer, config)

    def test_train_keyerror(self):

        model = CDAQtoNiDAQ(self.get_configs_NIDAQ())
        results = {}
        threshhold = 10000
        size = torch.randint(0, threshhold, (1, 1)).item()
        results["inputs"] = TorchUtils.format(torch.rand((size, 1)))
        results["wrong key"] = TorchUtils.format(torch.randint(
            0, 2, (size, 1)))
        results["norm_inputs"] = zscore_norm(results["inputs"])
        dataloader = get_data(results, batch_size=512)
        dataloaders = [dataloader]
        criterion = function.accuracy_fit
        optimizer = GeneticOptimizer(gene_ranges=TorchUtils.format(
            torch.tensor([[-1.2, 0.6], [-1.2, 0.6]])),
            partition=[4, 22],
            epochs=100)
        configs = {}
        configs["epochs"] = 100
        configs["stop_threshold"] = 0.5
        with self.assertRaises(AssertionError):
            train(model, dataloaders, criterion, optimizer, configs)

        results["targets"] = TorchUtils.format(torch.randint(0, 2, (size, 1)))
        configs = {}
        configs["wrong key"] = 100
        configs["stop_threshold"] = 0.5
        with self.assertRaises(AssertionError):
            train(model, dataloaders, criterion, optimizer, configs)

    def test_evaluate_population_CDAQ(self):

        # CDAQ
        try:
            model = CDAQtoCDAQ(self.get_configs_CDAQ())
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

    def test_evaluate_population_NIDAQ(self):

        # NIDAQ
        try:
            model = CDAQtoNiDAQ(self.get_configs_NIDAQ())
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

    def test_evaluate_population_fail(self):

        inputs = ["invalid type", 100, [1, 2, 3, 4, 5], {}]
        targets = ["invalid type", 100, [1, 2, 3, 4, 5], {}]
        dataloaders, criterion, optimizer, configs = self.get_train_parameters(
        )
        input, target = dataloaders[0].dataset[:]
        input, target = TorchUtils.format(input), TorchUtils.format(target)
        model = CDAQtoNiDAQ(self.get_configs_NIDAQ())
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
    unittest.main()
