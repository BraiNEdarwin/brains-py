import unittest
import numpy as np
import brainspy
from brainspy.processors.processor import Processor
from brainspy.processors.hardware.drivers.nidaq import CDAQtoNiDAQ


class NIDAQ_Processor_Test(unittest.TestCase):
    """
    Tests for the hardware processor with the CDAQ to NIDAQ driver.

    """
    def __init__(self, test_name):
        super(NIDAQ_Processor_Test, self).__init__()
        configs = {}
        configs["processor_type"] = "cdaq_to_nidaq"
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
        configs["driver"]["instruments_setup"][
            "readout_instrument"] = "cDAQ1Mod4"
        configs["driver"]["instruments_setup"]["readout_channels"] = [
            4
        ]  # Channels for reading the output current values

        configs["waveform"] = {}
        configs["waveform"]["plateau_length"] = 10
        configs["waveform"]["slope_length"] = 30

        self.configs = configs
        if brainspy.TEST_MODE == "HARDWARE_NIDAQ":
            self.model_data = {}
            self.model_data["info"] = {}
            self.model_data["info"]["sampling_configs"] = {
                "input_data": {
                    "amplitude": [0.9, 0.9, 0.9, 0.9, 0.9, 0.45, 0.5],
                    "batch_time": 50,
                    "device_no": 4,
                    "factor": 0.05,
                    "input_distribution": "sawtooth",
                    "input_electrodes": 7,
                    "input_frequency": [2, 3, 5, 7, 13, 17, 19],
                    "number_batches": 3500,
                    "offset": [-0.3, -0.3, -0.3, -0.3, -0.3, 0.15, -0.2],
                    "output_electrodes": 1,
                    "phase": [-0.57, 0.25, -1.54, 2.17, 0.08, 0.15, -0.65],
                    "ramp_time": 0.5,
                },
                "processor": {
                    "auto_start":
                    True,
                    "data": {
                        "input_electrode_no": 7,
                        "input_indices": [0, 1, 2, 3, 4, 5, 6],
                        "shape": 2550,
                        "waveform": {
                            "slope_length": 25.0
                        },
                    },
                    "driver": {
                        "amplification": 28.5,
                        "gain_info": "28.5MV/Afor1",
                        "instruments_setup": {
                            "activation_channel_mask": [1, 1, 1, 1, 1, 1, 1],
                            "activation_channels": [3, 4, 5, 6, 2, 1, 0],
                            "activation_instrument":
                            "cDAQ1Mod3",
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
                        "post_gain": 1,
                        "sampling_frequency": 50,
                        "tasks_driver_type": "local",
                    },
                    "electrode_setup": [
                        [
                            "ao8", "ao10", "ao13", "ao11", "ao7", "ao12",
                            "ao14", "out"
                        ],
                        [8, 10, 13, 11, 7, 12, 14],
                        [0, 1, 2, 3, 4, 5, 6],
                    ],
                    "max_ramping_time_seconds":
                    0.03,
                    "offset":
                    1,
                    "platform":
                    "hardware",
                    "processor_type":
                    "cdaq_to_cdaq",
                },
                "save_directory":
                "tmp/data/training/TEST\\Brains_testing_2021_02_15_134514",
            }
            self.model_data["info"]["electrode_info"] = {
                "electrode_no": 8,
                "activation_electrodes": {
                    "electrode_no":
                    7,
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

    @unittest.skipUnless(
        brainspy.TEST_MODE == "HARDWARE_NIDAQ",
        "Method deactivated as it is only possible to be tested on a NIDAQ TO NIDAQ setup",
    )
    def test_init(self):
        """
        Test to check correct initialization of the Hardware processor with the cdaq_to_cdaq driver.
        """
        isinstance(self.model.driver, CDAQtoNIDAQ)

    @unittest.skipUnless(
        brainspy.TEST_MODE == "HARDWARE_NIDAQ",
        "Method deactivated as it is only possible to be tested on a CDAQ TO NIDAQ setup",
    )
    def test_forward_numpy(self):
        """
        Test if a forward pass through the processor does not raise any exceptions.
        """
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        raised = False
        try:
            x = self.model.forward_numpy(x)
        except AssertionError:
            raised = True
        self.assertFalse(raised, "Exception raised")

    @unittest.skipUnless(
        brainspy.TEST_MODE == "HARDWARE_NIDAQ",
        "Method deactivated as it is only possible to be tested on a CDAQ TO NIDAQ setup",
    )
    def test_readout_trial(self):
        """
        Testing if the data can be read in the readout trial method
        """
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        data, finished = self.model.readout_trial(y)
        self.assertTrue(data is not None)

    @unittest.skipUnless(
        brainspy.TEST_MODE == "HARDWARE_NIDAQ",
        "Method deactivated as it is only possible to be tested on a CDAQ TO NIDAQ setup",
    )
    def test_synchronise_input_data(self):
        """
        Test to check if the input data returns synchronized data of the correct shape
        """
        y = np.array([1.0, 2.0], [3, 4])
        synchronized_data = self.model.synchronise_input_data(y)
        self.assertEqual(synchronized_data.shape, (2, 8))

    @unittest.skipUnless(
        brainspy.TEST_MODE == "HARDWARE_NIDAQ",
        "Method deactivated as it is only possible to be tested on a CDAQ TO NIDAQ setup",
    )
    def test_get_output_cut(self):
        """
        Test to check if the input produces the correct output cut value
        """
        y = np.array([(1, 2), (3, 4)])
        cut_val = self.model.get_output_cut_value(y)
        self.assertEqual(cut_val, 1)

    @unittest.skipUnless(
        brainspy.TEST_MODE == "HARDWARE_NIDAQ",
        "Method deactivated as it is only possible to be tested on a CDAQ TO NIDAQ setup",
    )
    def test_synchronise_output_data(self):
        """
        Test to check if output data is correctly synchronized based on the output cut value
        """
        y = np.array([(1, 2), (3, 4)])
        output = self.model.synchronize_onput_data(y)
        self.assertEqual(output, [[2]])

    def runTest(self):
        self.test_init()
        self.test_forward_numpy()
        self.test_readout_trial()
        self.test_synchronise_input_data()
        self.test_get_output_cut()
        self.test_synchronise_output_data()


if __name__ == "__main__":
    unittest.main()
