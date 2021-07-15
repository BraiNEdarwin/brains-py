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


class RemoteTasks_Test(unittest.TestCase):

    """
    Tests for remote tasks setup with custom configs and real time rack.
    """

    def __init__(self, test_name):
        super(RemoteTasks_Test, self).__init__()
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
        configs["driver"]["real_time_rack"] = True
        configs["driver"]["sampling_frequency"] = 1000
        configs["driver"]["instruments_setup"] = {}
        configs["driver"]["instruments_setup"]["multiple_devices"] = False
        configs["driver"]["instruments_setup"]["trigger_source"] = "cDAQ1/segment1"
        configs["driver"]["instruments_setup"]["activation_instrument"] = "cDAQ1Mod3"
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
        configs["driver"]["instruments_setup"]["readout_instrument"] = "cDAQ1Mod4"
        configs["driver"]["instruments_setup"]["readout_channels"] = [4]
        configs["waveform"] = {}
        configs["waveform"]["plateau_length"] = 10
        configs["waveform"]["slope_length"] = 30
        configs["driver"]["uri"] = "uri"
        configs["ip"] = None
        configs["port"] = None
        configs["subnet_mask"] = None
        configs["force_static_ip"] = None
        self.local = None
        self.configs = configs

    def test_get_driver(self):
        """
        Test to get a remoteTasks driver with a config file that has a real time rack set to true
        """
        tasks_driver = None
        try:
            tasks_driver = get_tasks_driver(self.configs["driver"])
        except:
            print("Invalid URI - only for testing this method ")
        isinstance(tasks_driver, RemoteTasks)

    def test_deploy_driver(self):
        try:
            deploy_driver(self.configs)
        except:
            print("Error")
        self.assertEqual(self.configs["ip"], "192.168.1.5")
        self.assertEqual(self.configs["subnet_mask"], "255.255.255.0")
        self.assertEqual(self.configs["port"], 8081)

    def runTest(self):
        self.test_get_driver()
        self.test_deploy_driver()


if __name__ == "__main__":
    unittest.main()
