import os
import torch
import unittest
import numpy as np
import brainspy
import nidaqmx
import nidaqmx.constants as constants
import nidaqmx.system.device as device
from brainspy.processors.processor import Processor
from brainspy.processors.hardware.drivers.cdaq import CDAQtoCDAQ
from brainspy.processors.hardware.drivers.ni.tasks import *


class Tasks_Test(unittest.TestCase):
    """
    Tests for tasks with custom configs and no real time rack.
    """
    def __init__(self, test_name):
        super(Tasks_Test, self).__init__()
        configs = {}
        configs["processor_type"] = "cdaq_to_cdaq"
        configs["input_indices"] = [2, 3]
        configs["electrode_effects"] = {}
        configs["electrode_effects"]["amplification"] = 3
        configs["electrode_effects"]["clipping_value"] = [-300, 300]
        configs["electrode_effects"]["noise"] = {}
        configs["electrode_effects"]["noise"]["noise_type"] = "gaussian"
        configs["electrode_effects"]["noise"]["variance"] = 0.6533523201942444
        configs["driver"] = {}
        configs["driver"]["sampling_frequency"] = 1000
        configs["driver"]["instruments_setup"] = {}
        configs["driver"]["instruments_setup"]["multiple_devices"] = False
        configs["driver"]["instruments_setup"][
            "trigger_source"] = "cDAQ1/segment1"
        configs["driver"]["instruments_setup"][
            "activation_instrument"] = "cDAQ1Mod3"
        configs["driver"]["instruments_setup"]["activation_channels"] = [
            0,
            2,
            5,
            3,
            4,
            6,
            1,
        ]
        configs["driver"]["instruments_setup"]["activation_voltages"] = [
            [-1.2, 0.6],
            [-1.2, 0.6],
            [-1.2, 0.6],
            [-1.2, 0.6],
            [-1.2, 0.6],
            [-0.7, 0.3],
            [-0.7, 0.3],
        ]
        configs["driver"]["instruments_setup"][
            "readout_instrument"] = "cDAQ1Mod4"
        configs["driver"]["instruments_setup"]["readout_channels"] = [4]
        configs["waveform"] = {}
        configs["waveform"]["plateau_length"] = 10
        configs["waveform"]["slope_length"] = 30
        self.configs = configs
        self.local = IOTasksManager()

    @unittest.skipIf(
        brainspy.TEST_MODE == "SIMULATION_PC",
        "Method deactivated as it is only possible to be tested on a CDAQ TO CDAQ setup",
    )
    def test_IOTasksManager_init(self):
        """
        Test to check IOTasksManager driver is initialized correctly.
        """
        self.assertEqual(self.local.acquisition_type,
                         constants.AcquisitionType.FINITE)
        self.assertEqual(self.local.activation_task, None)
        self.assertEqual(self.local.readout_task, None)

    @unittest.skipIf(
        brainspy.TEST_MODE == "SIMULATION_PC",
        "Method deactivated as it is only possible to be tested on a CDAQ TO CDAQ setup",
    )
    def test_init_tasks(self):
        """
        Test to initialize the tasks in the IOTasksManager driver
        """
        voltage_ranges = self.local.init_tasks(
            self.configs["driver"]
        )  # key error at channels.py "activation_voltage_ranges"
        self.assertEqual(voltage_ranges.shape, (7, 2))

    @unittest.skipIf(
        brainspy.TEST_MODE == "SIMULATION_PC",
        "Method deactivated as it is only possible to be tested on a CDAQ TO CDAQ setup",
    )
    def test_init_activation_channels(self):
        """
        Test to initialize the activation channels and check if the activation task is not None anymore
        """
        channel_names = [
            "cDAQ1Mod3/ao3",
            "cDAQ1Mod3/ao4",
            "cDAQ1Mod3/ao5",
            "cDAQ1Mod3/ao6",
            "cDAQ1Mod3/ao2",
            "cDAQ1Mod3/ao1",
            "cDAQ1Mod3/ao0",
        ]
        self.local.init_activation_channels(channel_names)
        self.assertTrue(self.local.activation_task is not None)

    @unittest.skipIf(
        brainspy.TEST_MODE == "SIMULATION_PC",
        "Method deactivated as it is only possible to be tested on a CDAQ TO CDAQ setup",
    )
    def test_init_readout_channels(self):
        """
        Test to initialize the readout channels and check if the readout task is not None anymore
        """
        channel_names = ["cDAQ1Mod4/ai4"]
        self.local.init_readout_channels(channel_names)
        self.assertTrue(self.local.readout_task is not None)

    @unittest.skipIf(
        brainspy.TEST_MODE == "SIMULATION_PC",
        "Method deactivated as it is only possible to be tested on a CDAQ TO CDAQ setup",
    )
    def test_close_tasks(self):
        """
        Test to close the IOTasksManager driver
        """
        self.local.close_tasks()
        self.assertEqual(self.local.activation_task, None)
        self.assertEqual(self.readout_task, None)

    @unittest.skipIf(
        brainspy.TEST_MODE == "SIMULATION_PC",
        "Method deactivated as it is only possible to be tested on a CDAQ TO CDAQ setup",
    )
    def test_read(self):
        """
        Test to read data using this driver and checking if any values are read
        """
        values = self.local.read((3, 3), 10)
        self.assertTrue(values is not None)

    @unittest.skipIf(
        brainspy.TEST_MODE == "SIMULATION_PC",
        "Method deactivated as it is only possible to be tested on a CDAQ TO CDAQ setup",
    )
    def test_remote_read(self):
        """
        Test to remotely read data using this driver and checking if any values are read
        """
        values = self.local.remote_read((3, 3), 10)
        self.assertTrue(values is not None)

    def runTest(self):
        self.test_get_driver()
        self.test_IOTasksManager_init()
        self.test_init_tasks()
        self.test_init_activation_channels()
        self.test_init_readout_channels()
        self.test_close_tasks()
        self.test_read()
        self.test_remote_read()


if __name__ == "__main__":
    unittest.main()
