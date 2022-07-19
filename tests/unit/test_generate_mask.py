"""
Module for testing waveform transformations.
"""
import torch
import unittest
from brainspy.utils.pytorch import TorchUtils
from brainspy.utils.waveform import WaveformManager


class WaveformTest(unittest.TestCase):
    """
    Class for testing the method - generate_mask() in waveform.py.
    """
    def test_generate_mask_manual(self):
        """
        Test to generate a mask.
        """
        configs = {"plateau_length": 2, "slope_length": 1}
        manager = WaveformManager(configs)
        output = TorchUtils.format(manager.generate_mask(7))
        self.assertTrue(
            torch.equal(
                output,
                torch.tensor([False, True, True, False, True, True, False],
                             device=TorchUtils.get_device(),
                             dtype=torch.bool)), "Generate mask error")

    def test_generate_mask_different_sizes(self):
        """
        Test to generate a mask with tensors of different sizes
        """
        configs = {}
        configs["plateau_length"] = 80
        configs["slope_length"] = 20
        waveform_mgr = WaveformManager(configs)
        test_sizes = ((1, 1), (10, 1), (100, 1), (10, 2), (100, 7))
        test_points = []
        for size in test_sizes:
            test_points.append(
                torch.rand(
                    size,
                    device=TorchUtils.get_device(),
                    dtype=torch.get_default_dtype(),
                ))
        for points in test_points:
            waveform = waveform_mgr.points_to_waveform(points)
            mask = waveform_mgr.generate_mask(len(waveform))
            self.assertEqual(len(mask), len(waveform),
                             "Generate mask - wrong size.")
            self.assertEqual(len(waveform[mask]),
                             waveform_mgr.plateau_length * len(points),
                             "Generate mask error")

    def test_generate_mask_slope_0(self):
        """
        Test to generate a mask with slope = 0
        """
        configs = {}
        configs["plateau_length"] = 10
        configs["slope_length"] = 0
        waveform = WaveformManager(configs)
        output = TorchUtils.format(waveform.generate_mask(10))
        self.assertTrue(
            torch.equal(
                output,
                torch.tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                             device=TorchUtils.get_device(),
                             dtype=torch.bool).float()), "Generate mask error")

    def test_generate_mask_plateau_0(self):
        """
        Test to generate a mask with plateau = 0
        """
        configs = {}
        configs["plateau_length"] = 0
        configs["slope_length"] = 10
        waveform = WaveformManager(configs)
        output = TorchUtils.format(waveform.generate_mask(10))
        self.assertTrue(
            torch.equal(
                output,
                torch.tensor([
                    False, False, False, False, False, False, False, False,
                    False, False
                ],
                             device=TorchUtils.get_device(),
                             dtype=torch.bool)), "Generate mask error")

    def test_generate_mask_slope_plateau_0(self):
        """
        Test to generate a mask with both slope and plateau = 0 raises ZeroDivisionError
        """
        configs = {}
        configs["plateau_length"] = 0
        configs["slope_length"] = 0
        waveform = WaveformManager(configs)
        with self.assertRaises(ZeroDivisionError):
            waveform.generate_mask(10)

    def test_generate_mask_slope_plateau_negative(self):
        """
        Test to generate a mask with negative plateau and/or slope
        returns an empty tensor
        """
        configs = {}
        configs["plateau_length"] = -10
        configs["slope_length"] = -10
        waveform = WaveformManager(configs)
        output = TorchUtils.format(waveform.generate_mask(10))
        self.assertTrue(
            torch.equal(
                output,
                torch.tensor([],
                             device=TorchUtils.get_device(),
                             dtype=torch.bool).float()), "Generate mask error")


if __name__ == "__main__":
    unittest.main()
