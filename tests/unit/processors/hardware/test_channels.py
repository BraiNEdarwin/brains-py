import torch
import unittest
import numpy as np
import brainspy
from brainspy.processors.hardware.drivers.ni.channels import (
    init_channel_data,
    init_activation_channels,
    init_readout_channels,
    get_mask,
    add_uniquely,
    concatenate_voltage_ranges,
)


class Channels_Test(unittest.TestCase):
    """
    Tests for the channels.py
    """

    def get_configs_multiple_devices(self):
        """
        Get the sample configurations for multiple devices:
        Activation instruments: cDAQ1Mod3,cDAQ1Mod2,cDAQ1Mod1
        Readout instrument: cDAQ1Mod4
        """
        configs = {}
        configs["multiple_devices"] = True
        configs["trigger_source"] = "cDAQ1/segment1"
        configs["A"] = {}
        configs["A"]["activation_instrument"] = "cDAQ1Mod3"
        configs["A"]["activation_channels"] = [
            0,
            2,
            5,
            3,
            4,
            6,
            1,
        ]
        configs["A"]["activation_voltage_ranges"] = [
            [-1.2, 0.6],
            [-1.2, 0.6],
            [-1.2, 0.6],
            [-1.2, 0.6],
            [-1.2, 0.6],
            [-0.7, 0.3],
            [-0.7, 0.3],
        ]
        configs["A"]["activation_channel_mask"] = [0, 0, 0, 0, 0, 0, 0]
        configs["A"]["readout_instrument"] = "cDAQ1Mod4"
        configs["A"]["readout_channels"] = [4]
        configs["B"] = {}
        configs["B"]["activation_instrument"] = "cDAQ1Mod2"
        configs["B"]["activation_channels"] = [
            0,
            2,
            5,
            3,
            4,
            6,
            1,
        ]
        configs["B"]["activation_voltage_ranges"] = [
            [-1.2, 0.6],
            [-1.2, 0.6],
            [-1.2, 0.6],
            [-1.2, 0.6],
            [-1.2, 0.6],
            [-0.7, 0.3],
            [-0.7, 0.3],
        ]
        configs["B"]["activation_channel_mask"] = [0, 0, 0, 0, 0, 0, 0]
        configs["B"]["readout_instrument"] = "cDAQ1Mod4"
        configs["B"]["readout_channels"] = [4]
        configs["C"] = {}
        configs["C"]["activation_instrument"] = "cDAQ1Mod1"
        configs["C"]["activation_channels"] = [
            0,
            2,
            5,
            3,
            4,
            6,
            1,
        ]
        configs["C"]["activation_voltage_ranges"] = [
            [-1.2, 0.6],
            [-1.2, 0.6],
            [-1.2, 0.6],
            [-1.2, 0.6],
            [-1.2, 0.6],
            [-0.7, 0.3],
            [-0.7, 0.3],
        ]
        configs["C"]["activation_channel_mask"] = [0, 0, 0, 0, 0, 0, 0]
        configs["C"]["readout_instrument"] = "cDAQ1Mod4"
        configs["C"]["readout_channels"] = [4]
        return configs

    def get_configs(self):
        """
        Get the sample configurations for a single device
        Activation instruments: cDAQ1Mod3
        Readout instrument: cDAQ1Mod4
        """
        configs = {}
        configs["multiple_devices"] = False
        configs["trigger_source"] = "cDAQ1/segment1"
        configs["activation_instrument"] = "cDAQ1Mod3"
        configs["activation_channels"] = [
            0,
            2,
            5,
            3,
            4,
            6,
            1,
        ]
        configs["activation_voltage_ranges"] = [
            [-1.2, 0.6],
            [-1.2, 0.6],
            [-1.2, 0.6],
            [-1.2, 0.6],
            [-1.2, 0.6],
            [-0.7, 0.3],
            [-0.7, 0.3],
        ]
        configs["activation_channel_mask"] = [0, 0, 0, 0, 0, 0, 0]
        configs["readout_instrument"] = "cDAQ1Mod4"
        configs["readout_channels"] = [4]
        return configs

    def test_init_channel_data_single_device(self):
        """
        Test to initialize activation channels,readout channels,instruments and voltage ranges for a single device
        """
        try:
            (
                activation_channel_list,
                readout_channel_list,
                instruments,
                voltage_ranges,
            ) = init_channel_data(self.get_configs())
            self.assertEqual(
                activation_channel_list,
                [
                    "cDAQ1Mod3/ao3",
                    "cDAQ1Mod3/ao4",
                    "cDAQ1Mod3/ao5",
                    "cDAQ1Mod3/ao6",
                    "cDAQ1Mod3/ao2",
                    "cDAQ1Mod3/ao1",
                    "cDAQ1Mod3/ao0",
                ],
            )
        except (Exception):
            self.fail("Could not initialize data")
        else:
            self.assertEqual(readout_channel_list, ["cDAQ1Mod4/ai4"])
            self.assertEqual(instruments, ["cDAQ1Mod3", "cDAQ1Mod4"])
            self.assertEqual(voltage_ranges.shape, (7, 2))

    def test_init_channel_data_multiple_devices(self):
        """
        Test to initialize activation channels,readout channels,instruments and voltage ranges for multiple devices
        """
        try:
            (
                activation_channel_list,
                readout_channel_list,
                instruments,
                voltage_ranges,
            ) = init_channel_data(self.get_configs_multiple_devices())
            self.assertEqual(
                activation_channel_list,
                [
                    "cDAQ1Mod3/ao3",
                    "cDAQ1Mod3/ao4",
                    "cDAQ1Mod3/ao5",
                    "cDAQ1Mod3/ao6",
                    "cDAQ1Mod3/ao2",
                    "cDAQ1Mod3/ao1",
                    "cDAQ1Mod3/ao0",
                ],
            )
        except (Exception):
            self.fail("Could not initialize data")
        else:
            self.assertEqual(readout_channel_list, ["cDAQ1Mod4/ai4"])
            self.assertEqual(instruments, ["cDAQ1Mod3", "cDAQ1Mod4"])
            self.assertEqual(voltage_ranges.shape, (7, 2))

    def test_init_channel_data_keyerror_single_device(self):

        with self.assertRaises(KeyError):
            configs = self.get_configs()
            del configs["activation_channel_mask"]
            init_channel_data(configs)

        with self.assertRaises(KeyError):
            configs = self.get_configs()
            del configs["readout_channels"]
            init_channel_data(configs)

        with self.assertRaises(KeyError):
            configs = self.get_configs()
            del configs["activation_channels"]
            init_channel_data(configs)

    def test_init_channel_data_keyerror_multiple_devices(self):

        with self.assertRaises(KeyError):
            configs = self.get_configs_multiple_devices()
            del configs["A"]["activation_channel_mask"]
            init_channel_data(configs)

        with self.assertRaises(KeyError):
            configs = self.get_configs_multiple_devices()
            del configs["B"]["readout_channels"]
            init_channel_data(configs)

        with self.assertRaises(KeyError):
            configs = self.get_configs_multiple_devices()
            del configs["C"]["activation_channels"]
            init_channel_data(configs)

    def test_concatenate_voltage_ranges(self):
        """
        Test to concatenate 2 voltage range lists
        """
        voltage_ranges1 = np.array([
            [-1.2, 0.6],
            [-1.2, 0.6],
            [-1.2, 0.6],
            [-1.2, 0.6],
            [-1.2, 0.6],
            [-0.7, 0.3],
            [-0.7, 0.3],
        ])
        voltage_ranges2 = np.array([
            [-1.4, 0.1],
            [-1.4, 0.1],
            [-1.4, 0.1],
            [-1.4, 0.1],
            [-1.4, 0.1],
            [-0.4, 0.1],
            [-0.4, 0.1],
        ])
        vlist = []
        vlist.append(voltage_ranges1)
        vlist.append(voltage_ranges2)
        v = concatenate_voltage_ranges(vlist)
        self.assertEqual(v.shape, (14, 2))

    def test_concatenate_voltage_ranges_fail(self):

        with self.assertRaises(AssertionError):
            concatenate_voltage_ranges([1, 2, 3, 4])
        with self.assertRaises(AssertionError):
            concatenate_voltage_ranges(100)
        with self.assertRaises(AssertionError):
            concatenate_voltage_ranges("invalid type")
        with self.assertRaises(AssertionError):
            concatenate_voltage_ranges({})

    def test_get_mask(self):
        """
        Test for the get mask method which returns None since it doset exist in the configs dictionary.
        """
        configs = self.get_configs()
        self.assertEqual(get_mask(configs),
                         np.array(configs["activation_channel_mask"]))

    def test_get_mask_fail(self):

        with self.assertRaises(AssertionError):
            get_mask([1, 2, 3])
        with self.assertRaises(AssertionError):
            get_mask(100)
        with self.assertRaises(AssertionError):
            get_mask("invalid type")

    def test_add_uniquely(self):
        """
        Test to add a unique element to an existing list of integers
        """
        mylist = [1, 2, 3, 4, 5]
        add_uniquely(mylist, 5)
        self.assertEqual(mylist, [1, 2, 3, 4, 5])
        add_uniquely(mylist, 6)
        self.assertEqual(mylist, [1, 2, 3, 4, 5, 6])

    def test_add_uniquely_fail(self):
        """
        Test to add a unique element to an existing list of integers
        """
        with self.assertRaises(AssertionError):
            add_uniquely({}, 5)
        with self.assertRaises(AssertionError):
            add_uniquely("invalid type", 5)
        with self.assertRaises(AssertionError):
            add_uniquely(np.array([1, 2, 3, 4]), 5)

    def test_init_activation_channels_single(self):

        try:
            a_list = init_activation_channels(self.get_configs())
        except (Exception):
            self.fail("Could not initialaize activation channels")
        else:
            self.assertTrue(a_list is not None and len(a_list > 0))

    def test_init_activation_channels_multiple(self):
        try:
            a_list = init_activation_channels(
                self.get_configs_multiple_devices())
        except (Exception):
            self.fail("Could not initialaize activation channels")
        else:
            self.assertTrue(a_list is not None and len(a_list > 0))

    def test_init_readout_channels_single(self):

        try:
            r_list = init_readout_channels(self.get_configs())
        except (Exception):
            self.fail("Could not initialaize readout channels")
        else:
            self.assertTrue(r_list is not None and len(r_list > 0))

    def test_init_readout_channels_multiple(self):
        try:
            r_list = init_readout_channels(self.get_configs_multiple_devices())
        except (Exception):
            self.fail("Could not initialaize readout channels")
        else:
            self.assertTrue(r_list is not None and len(r_list > 0))


if __name__ == "__main__":
    unittest.main()
