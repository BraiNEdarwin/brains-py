import unittest
import numpy as np
import random
import brainspy
from brainspy.processors.hardware.drivers.nidaq import CDAQtoNiDAQ


class NIDAQ_ReadoutTrial_Test(unittest.TestCase):
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
        "Method deactivated as it is only possible to be tested on a CDAQ TO CDAQ setup"
    )
    def test_readout_trial_random(self):
        """
        Test the readout_trial with random input data and check
        if the readout is not none
        """
        configs = self.get_configs()
        a1 = random.randint(1, 1000)
        a2 = len(configs["instruments_setup"]["activation_channels"]) + 1
        nidaq = CDAQtoNiDAQ(configs)
        y = np.random.rand(a1, a2) / 1000
        # Force them to start and end at zero
        y[0] = np.zeros_like(a2)
        y[-1] = np.zeros_like(a2)
        try:
            nidaq.original_shape = y.shape[0]
            val = nidaq.readout_trial(y.T)
        except (Exception):
            self.fail("Could not synchronise output data")
        finally:
            nidaq.close_tasks()
            self.assertIsNotNone(val)

    @unittest.skipUnless(
        brainspy.TEST_MODE == "HARDWARE_NIDAQ",
        "Method deactivated as it is only possible to be tested on a CDAQ TO CDAQ setup"
    )
    def test_readout_trial_invalid_type(self):
        """
        Invalid type for input raises an AssertionError
        """
        configs = self.get_configs()
        nidaq = CDAQtoNiDAQ(configs)
        with self.assertRaises(AssertionError):
            nidaq.readout_trial("Invalid type")
        with self.assertRaises(AssertionError):
            nidaq.readout_trial(500)
        with self.assertRaises(AssertionError):
            nidaq.readout_trial(100.10)
        with self.assertRaises(AssertionError):
            nidaq.readout_trial({"dict_key": 2})
        with self.assertRaises(AssertionError):
            nidaq.readout_trial([1, 2, 3, 4, 5, 6])
        nidaq.close_tasks()


if __name__ == "__main__":
    unittest.main(exit=False)
