import unittest
import warnings
import brainspy
from brainspy.processors.hardware.drivers.nidaq import CDAQtoNiDAQ


class NIDAQ_Init_Test(unittest.TestCase):
    """
    Test to intitalize the Nidaq Driver
    """
    def get_configs(self):
        """
        Generate configurations to initialize the Nidaq driver
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

    @unittest.skipUnless(
        brainspy.TEST_MODE == "HARDWARE_NIDAQ",
        "Method deactivated as it is only possible to be tested on a CDAQ TO NIDAQ setup"
    )
    def test_init(self):
        """
        Test to initialize the Nidaq driver
        """
        configs = self.get_configs()
        try:
            CDAQtoNiDAQ(configs)
        except (Exception):
            self.fail("Could not initialize the driver")

    @unittest.skipUnless(
        brainspy.TEST_MODE == "HARDWARE_NIDAQ",
        "Method deactivated as it is only possible to be tested on a CDAQ TO NIDAQ setup"
    )
    def test_init_keyerror(self):
        """
        Missing keys for the configurations of the NIdaq driver raises a key error
        """
        configs = self.get_configs()
        configs["instruments_setup"].pop("activation_channels", None)
        with self.assertRaises(KeyError):
            CDAQtoNiDAQ(configs)

    @unittest.skipUnless(
        brainspy.TEST_MODE == "HARDWARE_NIDAQ",
        "Method deactivated as it is only possible to be tested on a CDAQ TO NIDAQ setup"
    )
    def test_init_wrong_device_name_activation(self):
        """
        Incorrect name used for activation instrument raises
        a key error
        """
        configs = self.get_configs()
        configs["instruments_setup"][
            "activation_instrument"] = "ïnvalid device name"
        with self.assertRaises(KeyError):
            CDAQtoNiDAQ(configs)

    @unittest.skipUnless(
        brainspy.TEST_MODE == "HARDWARE_NIDAQ",
        "Method deactivated as it is only possible to be tested on a CDAQ TO NIDAQ setup"
    )
    def test_init_wrong_device_name_readout(self):
        """
        Incorrect name used for readout instrument raises
        a key error
        """
        configs = self.get_configs()
        configs["instruments_setup"][
            "readout_instrument"] = "ïnvalid device name"
        with self.assertRaises(KeyError):
            CDAQtoNiDAQ(configs)

    @unittest.skipUnless(
        brainspy.TEST_MODE == "HARDWARE_NIDAQ",
        "Method deactivated as it is only possible to be tested on a CDAQ TO NIDAQ setup"
    )
    def test_unequal_activation(self):
        """
        Unequal number of activation channels and activation voltage ranges raises assertion error
        """
        configs = self.get_configs()
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
        ]
        with self.assertRaises(AssertionError):
            CDAQtoNiDAQ(configs)

    @unittest.skipUnless(
        brainspy.TEST_MODE == "HARDWARE_NIDAQ",
        "Method deactivated as it is only possible to be tested on a CDAQ TO NIDAQ setup"
    )
    def test_very_high_low_voltages(self):
        """
        Very high or low activation volatges raises a warning
        """
        configs = self.get_configs()
        configs["instruments_setup"]["activation_voltage_ranges"] = [
            [-1.4, 0.7],
            [-1.2, 0.6],
            [-1.2, 0.6],
            [-1.2, 1.5],
            [-1.2, 0.6],
            [-0.7, 0.3],
            [-0.7, 0.3],
        ]
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            CDAQtoNiDAQ(configs)
            self.assertEqual(len(caught_warnings), 1)

    @unittest.skipUnless(
        brainspy.TEST_MODE == "HARDWARE_NIDAQ",
        "Method deactivated as it is only possible to be tested on a CDAQ TO NIDAQ setup"
    )
    def test_voltage_range_size(self):
        """
        Volatage range should contain a range with 2 values, otherwise
        an Assertion error is raised
        """
        configs = self.get_configs()
        configs["instruments_setup"]["activation_voltage_ranges"] = [[
            -1.4, 0.7
        ], [-1.2, 0.6], [-1.2, 0.6], [-1.2, 1.5], [-1.2, 0.6], [-0.7, 0.3],
                                                                     [-0.7]]

    @unittest.skipUnless(
        brainspy.TEST_MODE == "HARDWARE_NIDAQ",
        "Method deactivated as it is only possible to be tested on a CDAQ TO NIDAQ setup"
    )
    def test_sampling_val_error(self):
        """
        activation_sampling_frequency should be half of readout_sampling_frequency,
        otherwise an AssertionError is raised
        """
        configs = self.get_configs()
        configs["instruments_setup"]["activation_sampling_frequency"] = 500
        configs["instruments_setup"]["readout_sampling_frequency"] = 700
        with self.assertRaises(AssertionError):
            CDAQtoNiDAQ(configs)

        configs = self.get_configs()
        configs["instruments_setup"]["activation_sampling_frequency"] = 500
        configs["instruments_setup"]["readout_sampling_frequency"] = 1200
        with self.assertRaises(AssertionError):
            CDAQtoNiDAQ(configs)

    @unittest.skipUnless(
        brainspy.TEST_MODE == "HARDWARE_NIDAQ",
        "Method deactivated as it is only possible to be tested on a CDAQ TO NIDAQ setup"
    )
    def test_io_point_diff_false(self):
        """
        average_io_point_difference should be set to True for Nidaq,
        otherwise an assertion error is raised
        """
        configs = self.get_configs()
        configs["instruments_setup"]["average_io_point_difference"] = False
        with self.assertRaises(AssertionError):
            CDAQtoNiDAQ(configs)

    @unittest.skipUnless(
        brainspy.TEST_MODE == "HARDWARE_NIDAQ",
        "Method deactivated as it is only possible to be tested on a CDAQ TO NIDAQ setup"
    )
    def test_type_check(self):
        """
        Invalid type for configs raises an Assertion Error
        """
        configs = self.get_configs()
        configs["instruments_setup"]["average_io_point_difference"] = 100
        with self.assertRaises(AssertionError):
            CDAQtoNiDAQ(configs)

        configs = self.get_configs()
        configs["instruments_setup"]["activation_channels"] = True
        with self.assertRaises(AssertionError):
            CDAQtoNiDAQ(configs)

        configs = self.get_configs()
        configs["instruments_setup"][
            "readout_sampling_frequency"] = "Invalid type"
        with self.assertRaises(AssertionError):
            CDAQtoNiDAQ(configs)


if __name__ == "__main__":
    unittest.main()
