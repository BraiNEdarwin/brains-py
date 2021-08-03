import unittest

import torch
import numpy as np

from brainspy.processors.processor import Processor
from brainspy.utils.pytorch import TorchUtils


class ProcessorTest(unittest.TestCase):
    """
    Class for testing 'processor.py'.
    """
    def setUp(self):
        """
        Create different processors to run tests on.
        Tests init and load_processor methods.
        """
        electrode_effects = {
            "amplification": None,
            "output_clipping": [4.0, 3.0],
            "voltage_ranges": "default",
            "noise": None,
            "test": 0
        }
        electrode_info = {
            'activation_electrodes': {
                'electrode_no': 7,
                'voltage_ranges': [[1.0, 2.0]] * 7
            },
            'output_electrodes': {
                'electrode_no': 1,
                'amplification': [28.5],
                'clipping_value': [-114.0, 114.0]
            }
        }
        waveform = {"slope_length": 5, "plateau_length": 4}
        driver = {
            "instruments_setup": {
                "multiple_devices":
                False,
                "activation_instrument":
                "cDAQ1Mod3",
                "activation_channels": [8, 10, 13, 11, 7, 12, 14],
                "activation_voltage_ranges": [[-1.2, 0.6], [-1.2, 0.6],
                                              [-1.2, 0.6], [-1.2, 0.6],
                                              [-1.2, 0.6], [-0.7, 0.3],
                                              [-0.7, 0.3]],
                "readout_instrument":
                "cDAQ1Mod4",
                "readout_channels": [8, 10, 13, 11, 7, 12, 14],
                "trigger_source":
                "cDAQ1/segment1"
            },
            "real_time_rack": False,
            "sampling_frequency": 2,
            "output_clipping_range": [1.0, 2.0],
            "amplification": 3.0
        }
        # configs: sampling frequency
        configs_simulation = {
            "processor_type": "simulation",
            "electrode_effects": electrode_effects,
            "waveform": waveform,
            "driver": {}
        }
        configs_hardware = {
            "processor_type": "cdaq_to_cdaq",
            "electrode_effects": {},
            "waveform": waveform,
            "driver": driver
        }

        model_structure = {
            "D_in": 7,
            "D_out": 1,
            "activation": "relu",
            "hidden_sizes": [20, 20, 20]
        }
        info = {
            "model_structure": model_structure,
            "electrode_info": electrode_info
        }

        self.processor_simulation = Processor(configs_simulation, info)
        self.processor_hardware = Processor(configs_hardware, info)

    def test_load_processor(self):
        """
        Test if error is raised when processor type not recognized.
        """
        try:

            self.processor_simulation.load_processor(
                {"processor_type": "test"}, {})
            self.fail()
        except NotImplementedError:
            pass

    def test_forward(self):
        for i in range(100):
            x = TorchUtils.format(torch.rand(7))
            x = self.processor_simulation.forward(x)
            self.assertEqual(list(x.shape), [1])

    def test_format_targets(self):
        pass

    def test_get_voltage_ranges(self):
        pass

    def test_get_activation_electrode_no(self):
        pass

    def test_get_clipping_value(self):
        pass

    def test_swap(self):
        pass

    def test_is_hardware(self):
        pass

    def test_close(self):
        pass


if __name__ == "__main__":
    unittest.main()
