"""
Module for testing waveform transformations.
"""

import unittest
from brainspy.utils.waveform import WaveformManager


class WaveformTest(unittest.TestCase):
    """
    Class for testing the method - generate_mask_base() in waveform.py.
    """
    def test_generate_mask_base_pass(self):
        """
        Test to generate an initial and final mask for the torch tensor based
        on the configs of the waveform.
        """
        configs = {}
        configs["plateau_length"] = 80
        configs["slope_length"] = 20
        waveform = WaveformManager(configs)
        waveform.generate_mask_base()
        final_mask_list = waveform.final_mask.tolist()
        initial_mask_list = waveform.initial_mask.tolist()
        self.assertEqual(initial_mask_list, ([False] * 20) + ([True] * 80))
        self.assertEqual(final_mask_list, [False] * 20)

    def test_generate_mask_base_slope_0(self):
        """
        If slope length is 0, final mask list is empty
        """
        configs = {}
        configs["plateau_length"] = 10
        configs["slope_length"] = 0
        waveform = WaveformManager(configs)
        waveform.generate_mask_base()
        final_mask_list = waveform.final_mask.tolist()
        initial_mask_list = waveform.initial_mask.tolist()
        self.assertEqual(initial_mask_list, ([True] * 10))
        self.assertEqual(final_mask_list, [])

    def test_generate_mask_base_plateau_0(self):
        """
        If plateau length is 0,
        initial and final lists are equal
        """
        configs = {}
        configs["plateau_length"] = 0
        configs["slope_length"] = 10
        waveform = WaveformManager(configs)
        waveform.generate_mask_base()
        final_mask_list = waveform.final_mask.tolist()
        initial_mask_list = waveform.initial_mask.tolist()
        self.assertEqual(initial_mask_list, ([False] * 10))
        self.assertEqual(final_mask_list, [False] * 10)

    def test_generate_mask_base_slope_plateau_0(self):
        """
        If slope length and plateau length are both 0,
        initial and final lists are empty
        """
        configs = {}
        configs["plateau_length"] = 0
        configs["slope_length"] = 0
        waveform = WaveformManager(configs)
        waveform.generate_mask_base()
        final_mask_list = waveform.final_mask.tolist()
        initial_mask_list = waveform.initial_mask.tolist()
        self.assertEqual(initial_mask_list, [])
        self.assertEqual(final_mask_list, [])

    def test_generate_mask_base_slope_plateau_negative(self):
        """
        If slope length and plateau length are both negative,
        initial and final lists are empty.
        """
        configs = {}
        configs["plateau_length"] = -20
        configs["slope_length"] = -80
        waveform = WaveformManager(configs)
        waveform.generate_mask_base()
        final_mask_list = waveform.final_mask.tolist()
        initial_mask_list = waveform.initial_mask.tolist()
        self.assertEqual(initial_mask_list, [])
        self.assertEqual(final_mask_list, [])


if __name__ == "__main__":
    unittest.main()
