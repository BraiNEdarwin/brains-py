import unittest

import torch

import brainspy
from brainspy.utils.pytorch import TorchUtils
from brainspy.processors.processor import Processor


class ProcessorTest(unittest.TestCase):
    """
    Class for testing 'processor.py'.
    """
    def setUp(self):
        """
        Create different processors to run tests on.
        Tests init and load_processor methods.
        Assume 7 activation electrodes and 1 output electrode.
        """
        self.clipping = [4.0, 3.0]
        electrode_effects = {
            "amplification": None,
            "output_clipping": self.clipping,
            "voltage_ranges": "default",
            "noise": None,
            "test": 0
        }
        self.voltage = [[1.0, 2.0]] * 7
        electrode_info = {
            'activation_electrodes': {
                'electrode_no': 7,
                'voltage_ranges': self.voltage
            },
            'output_electrodes': {
                'electrode_no': 1,
                'amplification': [28.5],
                'clipping_value': [-114.0, 114.0]
            }
        }
        self.plateau = 6
        self.waveform = {"slope_length": 5, "plateau_length": self.plateau}
        self.configs_simulation = {
            "processor_type": "simulation",
            "electrode_effects": electrode_effects,
            "waveform": self.waveform,
            "driver": {}
        }

        model_structure = {
            "D_in": 7,
            "D_out": 1,
            "activation": "relu",
            "hidden_sizes": [20, 20, 20]
        }
        self.info = {
            "model_structure": model_structure,
            "electrode_info": electrode_info
        }

        self.processor_simulation = Processor(self.configs_simulation,
                                              self.info)
        self.processor_debug = Processor(self.configs_simulation, self.info)

    @unittest.skipIf(brainspy.TEST_MODE != "HARDWARE_CDAQ",
                     "Hardware test is skipped for simulation setup.")
    def test_cdaq(self):
        """
        Test the cdaq to cdaq processor type.
        """
        instruments_setup = {
            "activation_instrument":
            "cDAQ1Mod4",
            "activation_channels": [8, 10, 13, 11, 7, 12, 14],
            "activation_voltage_ranges":
            [[-1.2, 0.6], [-1.2, 0.6], [-1.2, 0.6], [-1.2, 0.6], [-1.2, 0.6],
             [-0.7, 0.3], [-0.7, 0.3]],
            "readout_instrument":
            "cDAQ1Mod3",
            "readout_channels": [2],
            "trigger_source":
            "cDAQ1/segment1"
        }
        driver = {
            "instruments_setup": instruments_setup,
            "real_time_rack": False,
            "sampling_frequency": 2,
            "output_clipping_range": [1.0, 2.0],
            "amplification": 1.0
        }
        configs_hardware = {
            "processor_type": "cdaq_to_cdaq",
            "electrode_effects": {},
            "waveform": self.waveform,
            "driver": driver
        }
        processor = Processor(configs_hardware, self.info)
        for i in range(1, 100):
            x = TorchUtils.format(torch.rand(i, 7))
            x = processor.forward(x)
            self.assertEqual(list(x.shape), [self.plateau * i, 1])
        self.processor.close()

    @unittest.skipIf(brainspy.TEST_MODE != "HARDWARE_NIDAQ",
                     "Hardware test is skipped for simulation setup.")
    def test_nidaq(self):
        """
        Test the cdaq to nidaq processor type.
        """
        instruments_setup = {
            "activation_instrument": "dev1",
            "activation_channels": [0, 1, 2, 3, 4, 5, 6],
            "readout_instrument": "cDAQ1Mod1",
            "readout_channels": [0]
        }
        driver = {
            "instruments_setup": instruments_setup,
            "real_time_rack": False,
            "sampling_frequency": 2,
            "output_clipping_range": [1.0, 2.0],
            "amplification": 1.0
        }
        configs_hardware = {
            "processor_type": "cdaq_to_nidaq",
            "electrode_effects": {},
            "waveform": self.waveform,
            "driver": driver
        }
        processor = Processor(configs_hardware, self.info)
        for i in range(1, 100):
            x = TorchUtils.format(torch.rand(i, 7))
            x = processor.forward(x)
            self.assertEqual(list(x.shape), [self.plateau * i, 1])
        self.processor.close()

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
        """
        Run forward pass and check shape of result. Takes into account the
        plateau length.
        """
        for i in range(1, 100):
            x = TorchUtils.format(torch.rand(i, 7))
            x = self.processor_simulation.forward(x)
            self.assertEqual(list(x.shape), [self.plateau * i, 1])
            x = TorchUtils.format(torch.rand(i, 7))
            x = self.processor_debug.forward(x)
            self.assertEqual(list(x.shape), [self.plateau * i, 1])

    def test_format_targets(self):
        """
        Check shape of data transformed to plateaus.
        """
        for i in range(1, 100):
            x = TorchUtils.format(torch.rand(i, 7))
            x = self.processor_simulation.format_targets(x)
            self.assertEqual(list(x.shape), [self.plateau * i, 7])
            x = TorchUtils.format(torch.rand(i, 7))
            x = self.processor_debug.format_targets(x)
            self.assertEqual(list(x.shape), [self.plateau * i, 7])

    def test_get_voltage_ranges(self):
        """
        Test the method for getting the voltage ranges. Compare to the list
        used to create the processor.
        """
        target = TorchUtils.format([[1.0, 2.0]] * 7)
        ranges = self.processor_simulation.get_voltage_ranges()
        self.assertTrue(torch.equal(ranges, target))
        ranges = self.processor_debug.get_voltage_ranges()
        self.assertTrue(torch.equal(ranges, target))

    def test_get_activation_electrode_no(self):
        """
        Test the method for getting the number of activation electrodes,
        should be 7.
        """
        self.assertEqual(
            self.processor_simulation.get_activation_electrode_no(), 7)
        self.assertEqual(self.processor_debug.get_activation_electrode_no(), 7)

    def test_get_clipping_value(self):
        """
        Test the method for getting the clipping value.
        """
        self.assertTrue(
            torch.equal(self.processor_simulation.get_clipping_value(),
                        TorchUtils.format(self.clipping)))
        self.assertTrue(
            torch.equal(self.processor_debug.get_clipping_value(),
                        TorchUtils.format(self.clipping)))

    def test_swap(self):
        """
        Test swap method.
        """
        self.processor_simulation.swap(self.configs_simulation, self.info)
        self.processor_debug.swap(self.configs_simulation, self.info)

    def test_is_hardware(self):
        """
        Test method for checking if processor is hardware.
        """
        self.assertFalse(self.processor_simulation.is_hardware())
        self.assertFalse(self.processor_debug.is_hardware())

    def test_close(self):
        """
        Test method for closing processor.
        """
        self.processor_simulation.close()
        self.processor_debug.close()


if __name__ == "__main__":
    unittest.main()
