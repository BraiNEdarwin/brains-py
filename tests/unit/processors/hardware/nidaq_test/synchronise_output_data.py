import unittest
import numpy as np
import random
import brainspy
from brainspy.processors.hardware.drivers.nidaq import CDAQtoNiDAQ


class NIDAQ_Output_Synchronise_Test(unittest.TestCase):
    """
    Test synchronise_output_data of the Nidaq Driver
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
    def test_synchronise_output_data(self):
        """
        Test to synchornise output data with random shape of data
        """
        a1 = random.randint(1, 1000)
        a2 = random.randint(1, 9)
        configs = self.get_configs()
        nidaq = CDAQtoNiDAQ(configs)
        y = np.random.rand(a1, a2)
        try:
            nidaq.original_shape = y.shape[0]
            val = nidaq.synchronise_output_data(y)
        except (Exception):
            self.fail("Could not synchronise output data")
        finally:
            self.assertIsNotNone(val)
            nidaq.close_tasks()

    @unittest.skipUnless(
        brainspy.TEST_MODE == "HARDWARE_NIDAQ",
        "Method deactivated as it is only possible to be tested on a CDAQ TO NIDAQ setup"
    )
    def test_synchronise_output_data_single_dimension(self):
        """
        Input data with single dimension raises an Index Error
        """
        a1 = random.randint(1, 1000)
        configs = self.get_configs()
        nidaq = CDAQtoNiDAQ(configs)
        y = np.random.rand(a1)
        with self.assertRaises(IndexError):
            nidaq.synchronise_output_data(y)
        nidaq.close_tasks()

    @unittest.skipUnless(
        brainspy.TEST_MODE == "HARDWARE_NIDAQ",
        "Method deactivated as it is only possible to be tested on a CDAQ TO NIDAQ setup"
    )
    def test_synchronise_invalid_type(self):
        """
        Invalid type for input raises a Type Error
        """
        configs = self.get_configs()
        nidaq = CDAQtoNiDAQ(configs)
        with self.assertRaises(AssertionError):
            nidaq.synchronise_output_data("Invalid type")
        with self.assertRaises(AssertionError):
            nidaq.synchronise_output_data(100)
        with self.assertRaises(AssertionError):
            nidaq.synchronise_output_data([1, 2, 3, 4])
        nidaq.close_tasks()


if __name__ == "__main__":
    unittest.main()
