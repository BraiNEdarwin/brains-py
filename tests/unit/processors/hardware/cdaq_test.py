import os
import torch
import unittest
import numpy as np
import brainspy
from brainspy.processors.processor import Processor
from brainspy.processors.hardware.drivers.cdaq import CDAQtoCDAQ


class CDAQ_Processor_Test(unittest.TestCase):

    """
    Tests for the hardware processor with the CDAQ to CDAQ driver.

    """

    # @unittest.skipUnless(brainspy.TEST_MODE == "HARDWARE_CDAQ", "Method deactivated as it is only possible to be tested on a CDAQ TO CDAQ setup")
    def __init__(self, test_name):
        super(CDAQ_Processor_Test, self).__init__()
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
        configs["driver"]["real_time_rack"] = False
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
        ]  # Channels through which voltages will be sent for activating the device (with both data inputs and control voltages)
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
        configs["driver"]["instruments_setup"]["readout_channels"] = [
            4
        ]  # Channels for reading the output current values

        configs["waveform"] = {}
        configs["waveform"]["plateau_length"] = 10
        configs["waveform"]["slope_length"] = 30

        self.configs = configs
        if brainspy.TEST_MODE == "HARDWARE_CDAQ":
            self.model_data = {}
            self.model_data["info"] = {}
            self.model_data["info"]["electrode_info"] = {
                "electrode_no": 8,
                "activation_electrodes": {
                    "electrode_no": 7,
                    "voltage_ranges": [
                        [-1.2, 0.6],
                        [-1.2, 0.6],
                        [-1.2, 0.6],
                        [-1.2, 0.6],
                        [-1.2, 0.6],
                        [-0.3, 0.6],
                        [-0.7, 0.3],
                    ],
                },
                "output_electrodes": {
                    "electrode_no": 1,
                    "amplification": 28.5,
                    "clipping_value": [-114.0, 114.0],
                },
            }
            self.model = Processor(
                self.configs,
                self.model_data["info"],
            )

    @unittest.skipIf(
        brainspy.TEST_MODE == "HARDWARE_CDAQ",
        "Method deactivated as it is only possible to be tested on a CDAQ TO CDAQ setup",
    )
    def test_init(self):
        """
        Test to check correct initialization of the Hardware processor with the cdaq_to_cdaq driver.
        """
        isinstance(self.model.driver, CDAQtoCDAQ)

    @unittest.skipIf(
        brainspy.TEST_MODE == "HARDWARE_CDAQ",
        "Method deactivated as it is only possible to be tested on a CDAQ TO CDAQ setup",
    )
    def test_forward_numpy(self):
        """
        Test if a forward pass through the processor returns a tensor of the
        right shape (the numpy version).
        """
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        x = self.model.forward_numpy(x)
        self.assertEqual(list(x.shape), [1])

    def runTest(self):
        self.test_init()
        self.test_forward_numpy()


if __name__ == "__main__":
    unittest.main()