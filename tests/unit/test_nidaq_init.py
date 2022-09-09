import unittest
import warnings
import brainspy
from brainspy.processors.hardware.drivers.ni.setup import NationalInstrumentsSetup
from tests.unit.testing_utils import check_test_configs
from brainspy.processors.hardware.drivers.nidaq import CDAQtoNiDAQ


class NIDAQ_Init_Test(unittest.TestCase):
    """
    Test init() of the Nidaq Driver.

    To run this file, the device has to be connected to a NIDAQ setup and
    the device configurations have to be specified depending on the setup.

    The test mode has to be set to HARDWARE_NIDAQ in tests/main.py.
    The required keys have to be defined in the get_configs() function.

    Some sample keys have been defined to run tests which do not require connection
    to the hardware.
    """
    def get_configs(self):
        """
        Generate configurations to initialize the Nidaq driver
        """
        configs = {}
        configs["inverted_output"] = True
        configs["amplification"] = 100
        configs["output_clipping_range"] = [-1, 1]

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
        return configs

    @unittest.skipUnless(
        brainspy.__TEST_MODE__ == "HARDWARE_NIDAQ",
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
        brainspy.__TEST_MODE__ == "HARDWARE_NIDAQ",
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
        brainspy.__TEST_MODE__ == "HARDWARE_NIDAQ",
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
        brainspy.__TEST_MODE__ == "HARDWARE_NIDAQ",
        "Method deactivated as it is only possible to be tested on a CDAQ TO NIDAQ setup"
    )
    def test_very_high_low_voltages(self):
        """
        Very high or low activation voltages raises a warning
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
        try:
            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always")
                CDAQtoNiDAQ(configs)
                self.assertEqual(len(caught_warnings), 1)
        except (Exception):
            self.fail()

    @unittest.skipUnless(
        brainspy.__TEST_MODE__ == "HARDWARE_NIDAQ",
        "Method deactivated as it is only possible to be tested on a CDAQ TO NIDAQ setup"
    )
    def test_voltage_range_size(self):
        """
        Voltage range should contain a range with 2 values, otherwise
        an IndexError is raised
        """
        configs = self.get_configs()
        configs["instruments_setup"]["activation_voltage_ranges"] = [[
            -1.4, 0.7
        ], [-1.2, 0.6], [-1.2, 0.6], [-1.2, 1.5], [-1.2, 0.6], [-0.7, 0.3],
                                                                     [-0.7]]
        with self.assertRaises(IndexError):
            CDAQtoNiDAQ(configs)

    @unittest.skipUnless(
        brainspy.__TEST_MODE__ == "HARDWARE_NIDAQ",
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
        brainspy.__TEST_MODE__ == "HARDWARE_NIDAQ",
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
        brainspy.__TEST_MODE__ == "HARDWARE_NIDAQ",
        "Method deactivated as it is only possible to be tested on a CDAQ TO NIDAQ setup"
    )
    def test_type_check(self):
        """
        Invalid type for configs raises an AssertionError or TypeError
        """
        configs = self.get_configs()
        configs["instruments_setup"]["average_io_point_difference"] = 100
        with self.assertRaises(AssertionError):
            CDAQtoNiDAQ(configs)

        configs = self.get_configs()
        configs["instruments_setup"]["activation_channels"] = True
        with self.assertRaises(TypeError):
            CDAQtoNiDAQ(configs)

        configs = self.get_configs()
        configs["instruments_setup"][
            "readout_sampling_frequency"] = "Invalid type"
        with self.assertRaises(AssertionError):
            CDAQtoNiDAQ(configs)


if __name__ == "__main__":
    testobj = NIDAQ_Init_Test()
    configs = testobj.get_configs()
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
