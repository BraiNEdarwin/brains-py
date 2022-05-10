import unittest
import numpy as np
import random
import brainspy
from brainspy.processors.hardware.drivers.nidaq import CDAQtoNiDAQ


class NIDAQ_Synchronise_Input_Test(unittest.TestCase):
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
    def test_synchronise_random_shape(self):
        """
        Test to synchronise input data for random shape of input
        """
        a1 = random.randint(1, 1000)
        a2 = random.randint(1, 9)
        configs = self.get_configs()
        nidaq = CDAQtoNiDAQ(configs)
        y = np.random.rand(a1, a2)
        try:
            new = nidaq.synchronise_input_data(y)
        except (Exception):
            self.fail("Could not synchronise inout data")
        test_synchronization_value = 0.04
        self.assertEqual(
            new.shape,
            (a1 + 1, a2 +
             (test_synchronization_value *
              configs["instruments_setup"]["activation_sampling_frequency"])))
        nidaq.close_tasks()

    # Failing test
    # @unittest.skipUnless(
    #     brainspy.TEST_MODE == "HARDWARE_NIDAQ",
    #     "Method deactivated as it is only possible to be tested on a CDAQ TO NIDAQ setup"
    # )
    # def test_synchronise_large_shape(self):

    #     configs = self.get_configs()
    #     nidaq = CDAQtoNiDAQ(configs)
    #     y = np.random.rand(1000, 4, 3, 2, 2)
    #     try:
    #         new = nidaq.synchronise_input_data(y)
    #     except(Exception):
    #         self.fail("Could not synchronise inout data")

    @unittest.skipUnless(
        brainspy.TEST_MODE == "HARDWARE_NIDAQ",
        "Method deactivated as it is only possible to be tested on a CDAQ TO NIDAQ setup"
    )
    def test_synchronise_single_dimension(self):
        """
        Test to synchronise input data with shape of only 1 dimension
        """
        for i in range(1, 10):
            configs = self.get_configs()
            nidaq = CDAQtoNiDAQ(configs)
            y = np.random.rand(i)
            try:
                nidaq.synchronise_input_data(y)
            except (Exception):
                self.fail("Could not synchronise inout data")
        nidaq.close_tasks()

    @unittest.skipUnless(
        brainspy.TEST_MODE == "HARDWARE_NIDAQ",
        "Method deactivated as it is only possible to be tested on a CDAQ TO NIDAQ setup"
    )
    def test_synchronise_output_contains_input(self):
        """
        Test to synchronise input data and check if output array
        contains all elements present in the input array
        """
        configs = self.get_configs()
        nidaq = CDAQtoNiDAQ(configs)
        y = np.random.rand(3, 2)
        try:
            new = nidaq.synchronise_input_data(y)
        except (Exception):
            self.fail("Could not synchronise inout data")

        mask = np.isin(y, new)
        check = np.all(mask)
        self.assertTrue(check)
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
            nidaq.synchronise_input_data("Invalid type")
        with self.assertRaises(AssertionError):
            nidaq.synchronise_input_data(100)
        nidaq.close_tasks()


if __name__ == "__main__":
    unittest.main()
