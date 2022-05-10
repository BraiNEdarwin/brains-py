import unittest
import numpy as np
import random
import warnings
import brainspy
from brainspy.processors.hardware.drivers.nidaq import CDAQtoNiDAQ


class NIDAQ_OutputCut_Test(unittest.TestCase):
    """
    Test synchronise_input_data of the Nidaq Driver
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
        brainspy.TEST_MODE == "HARDWARE_NIDAQ",
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
        brainspy.TEST_MODE == "HARDWARE_NIDAQ",
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
    unittest.main()
