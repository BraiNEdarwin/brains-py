"""
Module for testing waveform transformations.
"""
import unittest
import torch
from brainspy.utils.waveform import WaveformManager
from brainspy.utils.pytorch import TorchUtils


class WaveformTest(unittest.TestCase):
    """
    Class for testing the method - _init_() in waveform.py.
    """
    def setUp(self):
        """
        Generate some random datasets of different sizes.
        """
        configs = {}
        configs["plateau_length"] = 80
        configs["slope_length"] = 20
        self.configs = configs
        self.waveform_mgr = WaveformManager(configs)
        test_sizes = ((1, 1), (10, 1), (100, 1), (10, 2), (100, 7))
        self.test_points = []
        for size in test_sizes:
            self.test_points.append(
                torch.rand(
                    size,
                    device=TorchUtils.get_device(),
                    dtype=torch.get_default_dtype(),
                ))

    def test_init_pass(self):
        """
        Test to check initialization of variables from configs.
        """
        waveform = WaveformManager(self.configs)
        self.assertEqual(waveform.slope_length, 20)
        self.assertEqual(waveform.plateau_length, 80)

    def test_init_fail_slope_none(self):
        """
        NoneType slope length and plateau length raises Assertion error.
        """
        configs = {}
        configs["plateau_length"] = 80
        configs["slope_length"] = None
        with self.assertRaises(AssertionError):
            WaveformManager(configs)

    def test_init_fail_keyerror(self):
        """
        Missing keys in config file raises KeyError error.
        """
        configs = {}
        configs["plateau_length"] = 80
        with self.assertRaises(KeyError):
            WaveformManager(configs)

    def test_init_fail_none(self):
        """
        NoneType as an argument raises TypeError
        """
        configs = {}
        configs["plateau_length"] = 80
        with self.assertRaises(TypeError):
            WaveformManager(None)

    def test_init_fail_wrong_type(self):
        """
        Wrong type as an argument raises TypeError
        """
        configs = {}
        configs["plateau_length"] = 80
        with self.assertRaises(TypeError):
            WaveformManager([1, 2, 3, 4])


if __name__ == "__main__":
    unittest.main()
