import unittest
import torch
import brainspy
from brainspy.algorithms.gd import train, default_train_step, default_val_step
from brainspy.processors.dnpu import DNPU
from brainspy.processors.hardware.drivers.ni.setup import NationalInstrumentsSetup
from brainspy.processors.hardware.processor import HardwareProcessor
from brainspy.utils.pytorch import TorchUtils
from brainspy.utils.performance.accuracy import zscore_norm
from brainspy.utils.performance.data import get_data
from tests.unit.testing_utils import check_test_configs


class GD_Test(unittest.TestCase):
    """
    Tests for Gradient Descent

    To run this file, the device has to be connected to a NIDAQ setup and
    the device configurations have to be specified depending on the setup.

    The test mode has to be set to HARDWARE_NIDAQ in tests/main.py.
    The required keys have to be defined in the get_configs_NIDAQ() function.
    """
    def get_train_parameters(self):
        """
        Generate some random train parameters for Gradient Descent
        """
        configs = {}

        configs["inverted_output"] = True
        configs["amplification"] = 100
        configs["output_clipping_range"] = [-1, 1]

        configs["waveform"] = {}
        configs["waveform"]["plateau_length"] = 10
        configs["waveform"]["slope_length"] = 30

        configs["amplification"] = 100

        configs["instrument_type"] = "cdaq_to_nidaq"
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
        hp = HardwareProcessor(
            configs,
            slope_length=configs["waveform"]["slope_length"],
            plateau_length=configs["waveform"]["plateau_length"],
        )
        model = DNPU(hp, [[1, 2], [1, 3], [3, 4]])

        results = {}
        threshhold = 1000
        size = torch.randint(1, threshhold, (1, 1)).item()
        results["inputs"] = TorchUtils.format(torch.rand((size, 2)))
        results["targets"] = TorchUtils.format(torch.randint(0, 2, (size, 1)))
        results["norm_inputs"] = zscore_norm(results["inputs"])
        dataloader = get_data(results, batch_size=512)
        dataloaders = [dataloader]
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
        configs = {}
        configs["epochs"] = 100
        configs["constraint_control_voltages"] = "regul"
        configs["regul_factor"] = 0.5

        return model, dataloaders, criterion, optimizer, configs

    @unittest.skipUnless(brainspy.TEST_MODE == "HARDWARE_NIDAQ",
                         "Hardware test is skipped for simulation setup.")
    def test_train(self):
        """
        Test for genetic algorithm with random train parameters
        """
        model, dataloaders, criterion, optimizer, configs = self.get_train_parameters(
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

    @unittest.skipUnless(brainspy.TEST_MODE == "HARDWARE_NIDAQ",
                         "Hardware test is skipped for simulation setup.")
    def test_train_two_dataloaders(self):
        """
        Test for genetic algorithm with 2 dataloaders
        """
        model, dataloaders, criterion, optimizer, configs = self.get_train_parameters(
        )
        dataloaders.append(dataloaders[0])
        try:
            model, results = train(model, dataloaders, criterion, optimizer,
                                   configs)
        except Exception as e:
            print(e)

            self.fail("Could not run Gradient Descent: ")
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

    @unittest.skipUnless(brainspy.TEST_MODE == "HARDWARE_NIDAQ",
                         "Hardware test is skipped for simulation setup.")
    def test_invalid_model_type(self):
        """
        Wrong type of model raises an AssertionError
        """
        models = [{}, [1, 2, 3], "invalid type", 100, 5.5,
                  torch.nn.Linear(1, 1)]
        model, dataloaders, criterion, optimizer, configs = self.get_train_parameters(
        )
        for model in models:
            with self.assertRaises(AssertionError):
                train(model, dataloaders, criterion, optimizer, configs)
            with self.assertRaises(AssertionError):
                default_train_step(model, 10, dataloaders, criterion,
                                   optimizer)
            with self.assertRaises(AssertionError):
                default_val_step(10, model, dataloaders, criterion)

    @unittest.skipUnless(brainspy.TEST_MODE == "HARDWARE_NIDAQ",
                         "Hardware test is skipped for simulation setup.")
    def test_invalid_dataloader_type(self):
        """
        Wrong type of dataloader raises an AssertionError
        """
        model, dataloader, criterion, optimizer, configs = self.get_train_parameters(
        )
        dataloaders = [dataloader[0], {}, 100, "invalid type"]
        for d in dataloaders:
            with self.assertRaises(AssertionError):
                train(model, d, criterion, optimizer, configs)
        dataloaders2 = [dataloader, {}, 100, "invalid type"]
        for d in dataloaders2:
            with self.assertRaises(AssertionError):
                default_train_step(model, 10, d, criterion, optimizer)
            with self.assertRaises(AssertionError):
                default_val_step(10, model, d, criterion)

    @unittest.skipUnless(brainspy.TEST_MODE == "HARDWARE_NIDAQ",
                         "Hardware test is skipped for simulation setup.")
    def test_invalid_optimizer_type(self):
        """
        Wrong type of optimizer raises an AssertionError
        """
        model, dataloaders, criterion, optimizer, configs = self.get_train_parameters(
        )
        optimizers = ["invalid type", 100, [1, 2, 3, 4, 5], {}]
        for o in optimizers:
            with self.assertRaises(AssertionError):
                train(model, dataloaders, criterion, o, configs)
            with self.assertRaises(AssertionError):
                default_train_step(model, 10, dataloaders, criterion, o)

    @unittest.skipUnless(brainspy.TEST_MODE == "HARDWARE_NIDAQ",
                         "Hardware test is skipped for simulation setup.")
    def test_invalid_criterion_type(self):
        """
        Wrong type of criterion raises an AssertionError
        """
        model, dataloaders, criterion, optimizer, configs = self.get_train_parameters(
        )
        criterions = ["invalid type", 100, [1, 2, 3, 4, 5], {}]
        for c in criterions:
            with self.assertRaises(AssertionError):
                train(model, dataloaders, c, optimizer, configs)
            with self.assertRaises(AssertionError):
                default_train_step(model, 10, dataloaders, c, optimizer)
            with self.assertRaises(AssertionError):
                default_val_step(10, model, dataloaders, c)

    @unittest.skipUnless(brainspy.TEST_MODE == "HARDWARE_NIDAQ",
                         "Hardware test is skipped for simulation setup.")
    def test_train_invalid_configs_type(self):
        """
        Wrong type of configs raises an AssertionError
        """

        model, dataloaders, criterion, optimizer, configs = self.get_train_parameters(
        )
        configs = ["invalid type", 100, [1, 2, 3, 4, 5]]
        for config in configs:
            with self.assertRaises(AssertionError):
                train(model, dataloaders, criterion, optimizer, config)

    @unittest.skipUnless(brainspy.TEST_MODE == "HARDWARE_NIDAQ",
                         "Hardware test is skipped for simulation setup.")
    def test_default_train_step(self):
        try:
            model, dataloaders, criterion, optimizer, configs = self.get_train_parameters(
            )
            for epoch in range(0, 3):
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

    @unittest.skipUnless(brainspy.TEST_MODE == "HARDWARE_NIDAQ",
                         "Hardware test is skipped for simulation setup.")
    def test_default_val_step(self):
        try:
            model, dataloaders, criterion, optimizer, configs = self.get_train_parameters(
            )
            for epoch in range(0, 3):
                model, running_loss = default_train_step(epoch,
                                                         model,
                                                         dataloaders[0],
                                                         criterion,
                                                         optimizer,
                                                         logger=None)
                self.assertIsNotNone(running_loss)
                self.assertIsNone(model)
        except (Exception):
            self.fail("Could not perform default val step")


if __name__ == "__main__":
    testobj = GD_Test()
    if brainspy.TEST_MODE == "HARDWARE_NIDAQ":
        model, dataloaders, criterion, optimizer, configs = testobj.get_train_parameters(
        )
        try:
            NationalInstrumentsSetup.type_check(configs)
            if check_test_configs(configs):
                raise unittest.SkipTest(
                    "Configs are missing. Skipping all tests.")
            else:
                unittest.main()
        except (Exception):
            print(Exception)
            raise unittest.SkipTest(
                "Configs not specified correctly. Skipping all tests.")
    else:
        raise unittest.SkipTest(
            "Not connected to hardware nidaq. Skipping all tests.")
