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
    Tests for the hardware processor with the CDAQ to CDAQ driver.

    """
    def __init__(self, test_name):
        super(Channels_Test, self).__init__()
        self.configs = {
            "instruments_setup": {
                "multiple_devices":
                False,
                "activation_channels": [3, 4, 5, 6, 2, 1, 0],
                "activation_instrument":
                "cDAQ1Mod3",
                "activation_voltage_ranges": [
                    [-1.2, 0.6],
                    [-1.2, 0.6],
                    [-1.2, 0.6],
                    [-1.2, 0.6],
                    [-1.2, 0.6],
                    [-0.7, 0.3],
                    [-0.7, 0.3],
                ],
                "device_no":
                "single",
                "max_activation_voltages": [
                    0.6,
                    0.6,
                    0.6,
                    0.6,
                    0.6,
                    0.7,
                    0.3,
                ],
                "min_activation_voltages": [
                    -1.2,
                    -1.2,
                    -1.2,
                    -1.2,
                    -1.2,
                    -0.7,
                    -0.7,
                ],
                "readout_channels": [4],
                "readout_instrument":
                "cDAQ1Mod4",
                "trigger_source":
                "cDAQ1/segment1",
            },
        }

    @unittest.skipIf(
        brainspy.TEST_MODE == "SIMULATION_PC",
        "Method deactivated as it is only possible to be tested on a CDAQ TO CDAQ setup",
    )
    def test_init_channel_data(self):
        """
        Test to initialize activation channels,readout channels,instruments and voltage ranges
        """
        (
            activation_channel_list,
            readout_channel_list,
            instruments,
            voltage_ranges,
        ) = init_channel_data(self.configs)
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
        self.assertEqual(readout_channel_list, ["cDAQ1Mod4/ai4"])
        self.assertEqual(instruments, ["cDAQ1Mod3", "cDAQ1Mod4"])
        self.assertEqual(voltage_ranges.shape, (7, 2))

    @unittest.skipIf(
        brainspy.TEST_MODE == "SIMULATION_PC",
        "Method deactivated as it is only possible to be tested on a CDAQ TO CDAQ setup",
    )
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

    @unittest.skipIf(
        brainspy.TEST_MODE == "SIMULATION_PC",
        "Method deactivated as it is only possible to be tested on a CDAQ TO CDAQ setup",
    )
    def test_get_mask(self):
        """
        Test for the get mask method which returns None since it doset exist in the configs dictionary.
        """
        self.assertEqual(get_mask(self.configs), None)

    @unittest.skipIf(
        brainspy.TEST_MODE == "SIMULATION_PC",
        "Method deactivated as it is only possible to be tested on a CDAQ TO CDAQ setup",
    )
    def test_add_uniquely(self):
        """
        Test to add a unique element to an existing list of integers
        """
        mylist = [1, 2, 3, 4, 5]
        add_uniquely(mylist, 5)
        self.assertEqual(mylist, [1, 2, 3, 4, 5])
        add_uniquely(mylist, 6)
        self.assertEqual(mylist, [1, 2, 3, 4, 5, 6])

    def runTest(self):
        self.test_init_channel_data()
        self.test_concatenate_voltage_ranges()
        self.test_get_mask()
        self.test_add_uniquely()


if __name__ == "__main__":
    unittest.main()
