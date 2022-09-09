import unittest
import numpy as np
import random
import warnings
import brainspy
from brainspy.processors.hardware.drivers.ni.setup import NationalInstrumentsSetup
from utils import check_test_configs
from brainspy.processors.hardware.drivers.nidaq import CDAQtoNiDAQ


class NIDAQ_OutputCut_Test(unittest.TestCase):
    """
    Test output_cut_value() of the Nidaq Driver.

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
    def test_get_output_cut_random(self):
        """
        Test to get output cut value with random shape of data
        """
        a1 = random.randint(1, 1000)
        a2 = random.randint(1, 9)
        configs = self.get_configs()
        nidaq = CDAQtoNiDAQ(configs)
        y = np.random.rand(a1, a2)
        try:
            cut_val = nidaq.get_output_cut_value(y)
            self.assertIsNotNone(cut_val)
        except (Exception):
            self.fail("Could not get output cut value")
        finally:
            nidaq.close_tasks()

    @unittest.skipUnless(
        brainspy.__TEST_MODE__ == "HARDWARE_NIDAQ",
        "Method deactivated as it is only possible to be tested on a CDAQ TO NIDAQ setup"
    )
    def test_get_output_cut_low(self):
        """
        Test to get output cut value with cut value less that 0.05
        raises a "Spike not generated" warning
        """
        configs = self.get_configs()
        nidaq = CDAQtoNiDAQ(configs)
        y = np.array([[0.04, 0.04], [0.03, 0.03]])
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            try:
                nidaq.get_output_cut_value(y)
            except (Exception):
                self.fail("Could not get output cut value")
            finally:
                nidaq.close_tasks()
            self.assertEqual(len(caught_warnings), 1)

    @unittest.skipUnless(
        brainspy.__TEST_MODE__ == "HARDWARE_NIDAQ",
        "Method deactivated as it is only possible to be tested on a CDAQ TO NIDAQ setup"
    )
    def test_get_output_cut_invalid_type(self):
        """
        Invalid type for read_data raises an Assertion Error
        """
        configs = self.get_configs()
        nidaq = CDAQtoNiDAQ(configs)
        with self.assertRaises(AssertionError):
            nidaq.get_output_cut_value("Ïnvalid type")
        with self.assertRaises(AssertionError):
            nidaq.get_output_cut_value(100)
        with self.assertRaises(AssertionError):
            nidaq.get_output_cut_value([1, 2, 3, 4])
        nidaq.close_tasks()


if __name__ == "__main__":
    testobj = NIDAQ_OutputCut_Test()
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
