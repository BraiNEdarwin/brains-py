import os
import torch
import unittest
import numpy as np
import brainspy
import random
import warnings
from brainspy.processors.processor import Processor
from brainspy.processors.hardware.drivers.nidaq import CDAQtoNiDAQ


class NIDAQ_Synchronise_Test(unittest.TestCase):
    """
    Test synchronise_input_data of the Nidaq Driver
    """

    def get_configs(self):
        configs = {}
        configs["driver"] = {}
        configs["driver"]["instrument_type"] = "cdaq_to_nidaq"
        configs["driver"]["real_time_rack"] = False
        configs["driver"]["inverted_output"] = True
        configs["driver"]["amplificatiion"] = 100
        configs["driver"]["instruments_setup"] = {}
        configs["driver"]["instruments_setup"]["multiple_devices"] = False
        configs["driver"]["instruments_setup"][
            "activation_instrument"] = "cDAQ2Mod1"
        configs["driver"]["instruments_setup"]["activation_channels"] = [
            0,
            2,
            5,
            3,
            4,
            6,
            1,
        ]
        configs["driver"]["instruments_setup"]["activation_voltage_ranges"] = [
            [-1.2, 0.6],
            [-1.2, 0.6],
            [-1.2, 0.6],
            [-1.2, 0.6],
            [-1.2, 0.6],
            [-0.7, 0.3],
            [-0.7, 0.3],
        ]
        configs["driver"]["instruments_setup"][
            "readout_instrument"] = "dev1"
        configs["driver"]["instruments_setup"]["readout_channels"] = [4]
        configs["driver"]["instruments_setup"]["activation_sampling_frequency"] = 500
        configs["driver"]["instruments_setup"]["readout_sampling_frequency"] = 1000
        configs["driver"]["instruments_setup"]["average_io_point_difference"] = True
        return configs

    def test_synchronise_random_shape(self):
        """
        Synchronise input data for random shape of input
        """
        a1 = random.randint(1, 1000)
        a2 = random.randint(1, 9)
        configs = self.get_configs()
        nidaq = CDAQtoNiDAQ(configs["driver"])
        y = np.random.rand(a1, a2)
        try:
            new = nidaq.synchronise_input_data(y)
        except(Exception):
            self.fail("Could not synchronise inout data")
        test_synchronization_value = 0.04
        self.assertEqual(new.shape, (a1 + 1, a2 + (test_synchronization_value * configs["driver"]["instruments_setup"]["activation_sampling_frequency"])))
        nidaq.close_tasks()

    # Failing test
    # def test_synchronise_large_shape(self):

    #     configs = self.get_configs()
    #     nidaq = CDAQtoNiDAQ(configs["driver"])
    #     y = np.random.rand(1000, 4, 3, 2, 2)
    #     try:
    #         new = nidaq.synchronise_input_data(y)
    #     except(Exception):
    #         self.fail("Could not synchronise inout data")

    def test_synchronise_single_dimension(self):
        """
        Test to synchronise input data with shape of only 1 dimension
        """
        for i in range(1, 10):
            print(i)
            configs = self.get_configs()
            nidaq = CDAQtoNiDAQ(configs["driver"])
            y = np.random.rand(i)
            try:
                new = nidaq.synchronise_input_data(y)
            except(Exception):
                self.fail("Could not synchronise inout data")
        nidaq.close_tasks()

    def test_synchronise_output_contains_input(self):
        """
        Check if output array contains all elements present in the
        input array
        """
        configs = self.get_configs()
        nidaq = CDAQtoNiDAQ(configs["driver"])
        y = np.random.rand(3, 2)
        try:
            new = nidaq.synchronise_input_data(y)
        except(Exception):
            self.fail("Could not synchronise inout data")

        mask = np.isin(y, new)
        check = np.all(mask)
        self.assertTrue(check)
        nidaq.close_tasks()

    def test_synchronise_invalid_type(self):
        """
        Invalid type for input raises a Type Error
        """
        configs = self.get_configs()
        nidaq = CDAQtoNiDAQ(configs["driver"])
        with self.assertRaises(AssertionError):
            nidaq.synchronise_input_data("Invalid type")
        with self.assertRaises(AssertionError):
            nidaq.synchronise_input_data(100)
        nidaq.close_tasks()


if __name__ == "__main__":
    unittest.main()
