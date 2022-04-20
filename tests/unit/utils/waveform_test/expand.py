"""
Module for testing waveform transformations.
"""
import unittest
from brainspy.utils.waveform import WaveformManager


class WaveformTest(unittest.TestCase):
    """
    Class for testing the method - _expand() in waveform.py.
    """
    def setUp(self):
        """
        Generate some random datasets of different sizes.
        """
        configs = {}
        configs["plateau_length"] = 80
        configs["slope_length"] = 20
        self.configs = configs

    def test_expand(self):
        """
        Test to format the amplitudes and slopes to have the same length.
        """
        waveform = WaveformManager(self.configs)
        plateau_lengths = waveform._expand(waveform.plateau_length, 100)
        self.assertEqual(len(plateau_lengths), 100)

    def test_expand_worng_length(self):
        """
        Cannot expand with a wrong type length
        """
        waveform = WaveformManager(self.configs)
        with self.assertRaises(TypeError):
            waveform._expand(waveform.plateau_length, "worng input")

    def test_expand_nonetype_length(self):
        """
        Cannot expand with a NoneType length
        """
        waveform = WaveformManager(self.configs)
        with self.assertRaises(AssertionError):
            waveform._expand(waveform.plateau_length, None)

    def test_expand_none_param(self):
        """
        NoneType param return None on expanding
        """
        waveform = WaveformManager(self.configs)
        with self.assertRaises(AssertionError):
            waveform._expand(None, 100)

    def test_expand_extreme(self):
        """
        Test to format the amplitudes and slopes to have the same length with extreme values.
        """
        waveform = WaveformManager(self.configs)
        plateau_lengths = waveform._expand(10, 0)
        self.assertEqual(len(plateau_lengths), 0)

    def test_expand_negative_plateau_slope(self):
        """
        Test to expand with negative slopes or plateau raises no errors
        """
        configs = {}
        configs["plateau_length"] = -80
        configs["slope_length"] = -20
        waveform = WaveformManager(configs)
        try:
            waveform._expand(waveform.plateau_length, 100)
        except (Exception):
            self.fail("Exception Raised")


if __name__ == "__main__":
    unittest.main()
