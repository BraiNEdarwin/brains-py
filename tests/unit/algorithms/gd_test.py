import unittest
import torch
import brainspy.algorithms.modules.signal as function
from brainspy.algorithms.gd import train, default_train_step, default_val_step
from brainspy.utils.pytorch import TorchUtils
from brainspy.algorithms.modules.performance.accuracy import zscore_norm
from brainspy.algorithms.modules.performance.data import get_data
from brainspy.algorithms.modules.optim import GeneticOptimizer
from brainspy.processors.hardware.drivers.nidaq import CDAQtoNiDAQ
from brainspy.processors.hardware.drivers.cdaq import CDAQtoCDAQ


class GD_Test(unittest.TestCase):
    """
    Tests for the hardware processor with the CDAQ to CDAQ driver.
    """

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
        configs["constraint_control_voltages"] = "regul"
        configs["regul_factor"] = 0.5
        return dataloaders, criterion, optimizer, configs

    def get_configs_NIDAQ(self):
        """
        Generate configurations to initialize the cdaq driver
        """
        configs = {}
        configs["instrument_type"] = "cdaq_to_nidaq"
        configs["real_time_rack"] = False
        configs["inverted_output"] = True
        configs["amplification"] = 100
        configs["instruments_setup"] = {}
        configs["instruments_setup"]["multiple_devices"] = False
        configs["instruments_setup"]["activation_instrument"] = "cDAQ2Mod1"
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
        configs["instruments_setup"]["readout_instrument"] = "dev1"
        configs["instruments_setup"]["readout_channels"] = [4]
        configs["instruments_setup"]["activation_sampling_frequency"] = 500
        configs["instruments_setup"]["readout_sampling_frequency"] = 1000
        configs["instruments_setup"]["average_io_point_difference"] = True
        return configs

    def get_configs_CDAQ(self):
        """
        Generate configurations to initialize the Nidaq driver
        """
        configs = {}
        configs["instrument_type"] = "cdaq_to_cdaq"
        configs["real_time_rack"] = False
        configs["inverted_output"] = True
        configs["amplification"] = 100
        configs["instruments_setup"] = {}
        configs["instruments_setup"]["multiple_devices"] = False
        configs["instruments_setup"]["activation_instrument"] = "cDAQ2Mod1"
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
        configs["instruments_setup"]["readout_instrument"] = "dev1"
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
            self.fail("Could not run Gradient Descent")
        else:
            self.assertTrue("performance_history" in results)

        configs = {}
        configs["epochs"] = 100
        configs["constraint_control_voltages"] = "clip"
        try:
            model, results = train(model, dataloaders, criterion, optimizer,
                                   configs)
        except (Exception):
            self.fail("Could not run Gradient Descent")
        else:
            self.assertTrue("performance_history" in results)

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
            self.fail("Could not run Gradient Descent")
        else:
            self.assertTrue("performance_history" in results)

        configs = {}
        configs["epochs"] = 100
        configs["constraint_control_voltages"] = "clip"
        try:
            model, results = train(model, dataloaders, criterion, optimizer,
                                   configs)
        except (Exception):
            self.fail("Could not run Gradient Descent")
        else:
            self.assertTrue("performance_history" in results)

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
        configs["constraint_control_voltages"] = "regul"
        configs["regul_factor"] = 0.5
        with self.assertRaises(AssertionError):
            train(model, dataloaders, criterion, optimizer, configs)

        results["targets"] = TorchUtils.format(torch.randint(0, 2, (size, 1)))
        configs = {}
        configs["epochs"] = 100
        configs["constraint_control_voltages"] = "regul"
        configs["regul_factor"] = 0.5
        with self.assertRaises(AssertionError):
            train(model, dataloaders, criterion, optimizer, configs)

    def test_default_train_step_NIDAQ(self):
        try:
            model = CDAQtoNiDAQ(self.get_configs_NIDAQ())
            dataloaders, criterion, optimizer, configs = self.get_train_parameters(
            )
            for epoch in range(0, 100):
                model, running_loss = default_train_step(
                    model,
                    epoch,
                    dataloaders[0],
                    criterion,
                    optimizer,
                    logger=None,
                    constraint_control_voltages=configs[
                        'constraint_control_voltages'])
                self.assertIsNotNone(running_loss)
                self.assertIsNone(model)
        except (Exception):
            self.fail("Could not perform default train step")

    def test_default_val_step_NIDAQ(self):
        try:
            model = CDAQtoNiDAQ(self.get_configs_NIDAQ())
            dataloaders, criterion, optimizer, configs = self.get_train_parameters(
            )
            for epoch in range(0, 100):
                model, running_loss = default_train_step(epoch,
                                                         model,
                                                         dataloaders[0],
                                                         criterion,
                                                         logger=None)
                self.assertIsNotNone(running_loss)
                self.assertIsNone(model)
        except (Exception):
            self.fail("Could not perform default val step")

    def test_default_train_step_CDAQ(self):
        try:
            model = CDAQtoCDAQ(self.get_configs_CDAQ())
            dataloaders, criterion, optimizer, configs = self.get_train_parameters(
            )
            for epoch in range(0, 100):
                model, running_loss = default_train_step(
                    model,
                    epoch,
                    dataloaders[0],
                    criterion,
                    optimizer,
                    logger=None,
                    constraint_control_voltages=configs[
                        'constraint_control_voltages'])
                self.assertIsNotNone(running_loss)
                self.assertIsNone(model)
        except (Exception):
            self.fail("Could not perform default train step")

    def test_default_val_step_CDAQ(self):
        try:
            model = CDAQtoCDAQ(self.get_configs_CDAQ())
            dataloaders, criterion, optimizer, configs = self.get_train_parameters(
            )
            for epoch in range(0, 100):
                model, running_loss = default_train_step(epoch,
                                                         model,
                                                         dataloaders[0],
                                                         criterion,
                                                         logger=None)
                self.assertIsNotNone(running_loss)
                self.assertIsNone(model)
        except (Exception):
            self.fail("Could not perform default val step")


if __name__ == "__main__":
    unittest.main()
