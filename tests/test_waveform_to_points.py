"""
Module for testing waveform transformations.
"""
import unittest
import torch
import warnings
import numpy as np
from brainspy.utils.waveform import WaveformManager
from brainspy.utils.pytorch import TorchUtils


class WaveformTest(unittest.TestCase):
    """
    Class for testing the method - waveform_to_points() in waveform.py.
    """
    def test_waveform_to_points_manual(self):
        """
        Test the transform from waveform to points.
        """
        manager = WaveformManager({"plateau_length": 1, "slope_length": 2})
        data = torch.tensor([[0], [1], [1], [1], [5], [5], [5], [0]],
                            device=TorchUtils.get_device(),
                            dtype=torch.get_default_dtype())
        output = manager.waveform_to_points(data)
        self.assertTrue(
            torch.equal(
                output,
                torch.tensor([[1], [5]],
                             device=TorchUtils.get_device(),
                             dtype=torch.get_default_dtype())),
            "Waveform to points error")

    def test_waveform_to_points_different_sizes(self):
        """
        Test to generate points from a waveform with different sizes of tensors
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
            points_reverse = waveform_mgr.waveform_to_points(waveform)
            self.assertTrue(torch.allclose(points, points_reverse),
                            "Waveform to points error")

    def test_waveform_to_points_slope_plateau_0(self):
        """
        Test to generate points from a waveform with slope length = 0 raises mask IndexError
        """
        configs = {}
        configs["plateau_length"] = 80
        configs["slope_length"] = 20
        waveform_mgr = WaveformManager(configs)
        waveform = torch.rand((1, 1),
                              device=TorchUtils.get_device(),
                              dtype=torch.get_default_dtype())
        with self.assertRaises(IndexError):
            waveform_mgr.waveform_to_points(waveform)
        """
        Test to generate points from a waveform with plateau length = 0 raises ZeroDivisionError
        """
        configs = {}
        configs["plateau_length"] = 0
        configs["slope_length"] = 20
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            waveform_mgr = WaveformManager(configs)
            self.assertEqual(len(caught_warnings), 1)
        points = torch.rand(2, 2)
        waveform = waveform_mgr.points_to_waveform(
            points.to(TorchUtils.get_device()))
        with self.assertRaises(ZeroDivisionError):
            waveform_mgr.waveform_to_points(waveform)
        """
        Test to generate points from a waveform with both slope and plateau length = 0
        raises ZeroDivisionError
        """
        configs = {}
        configs["plateau_length"] = 0
        configs["slope_length"] = 0
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            waveform_mgr = WaveformManager(configs)
            self.assertEqual(len(caught_warnings), 2)
        points = torch.rand(2, 2)
        waveform = waveform_mgr.points_to_waveform(
            points.to(TorchUtils.get_device()))
        with self.assertRaises(ZeroDivisionError):
            waveform_mgr.waveform_to_points(waveform)

    def test_waveform_to_points_negative(self):
        """
        Test to generate points from a waveform with negative values raises no errors
        """
        configs = {}
        configs["plateau_length"] = 80
        configs["slope_length"] = 20
        waveform_mgr = WaveformManager(configs)
        points = torch.tensor([[1., -1.], [1., -1.]])
        try:
            waveform = waveform_mgr.points_to_waveform(
                points.to(TorchUtils.get_device()))
            waveform_mgr.waveform_to_points(waveform)
        except Exception:
            self.fail("Exception raised")

    def test_waveform_to_points_nonetype_tensor(self):
        """
        Test to generate points from a waveform with tensor of a NoneType value which raises
        RuntimeError
        """
        configs = {}
        configs["plateau_length"] = 1
        configs["slope_length"] = 10
        waveform_mgr = WaveformManager(configs)
        with self.assertRaises(RuntimeError):
            waveform_mgr.waveform_to_points(
                torch.tensor(None).to(TorchUtils.get_device()))

    def test_waveform_to_points_empty_tensor(self):
        """
        Test to generate points from a waveform with an empty tensor which raises no errors
        """
        configs = {}
        configs["plateau_length"] = 80
        configs["slope_length"] = 20
        waveform_mgr = WaveformManager(configs)
        points = torch.empty(1, 1)
        try:
            waveform = waveform_mgr.points_to_waveform(
                points.to(TorchUtils.get_device()))
            waveform_mgr.waveform_to_points(waveform)
        except Exception:
            self.fail("Exception raised")

    def test_waveform_to_points_single_val(self):
        """
        Test to generate points from a waveform with a tensor containing a single value which
        raises no errors
        """
        configs = {}
        configs["plateau_length"] = 80
        configs["slope_length"] = 20
        waveform_mgr = WaveformManager(configs)
        points = torch.tensor([1], dtype=torch.float32)
        with self.assertRaises(AssertionError):
            waveform = waveform_mgr.points_to_waveform(
                points.to(TorchUtils.get_device()))

    def test_waveform_to_points_invalid_type(self):
        """
        Test to generate points from a waveform with invalid type raises TypeError
        """
        configs = {}
        configs["plateau_length"] = 1
        configs["slope_length"] = 10
        waveform_mgr = WaveformManager(configs)
        with self.assertRaises(AssertionError):
            waveform_mgr.waveform_to_points([1, 2, 3, 4])
        with self.assertRaises(AssertionError):
            waveform_mgr.waveform_to_points(np.array(1))
        with self.assertRaises(AssertionError):
            waveform_mgr.waveform_to_points("String type is not accepted")
        with self.assertRaises(AssertionError):
            waveform_mgr.waveform_to_points(None)

    def test_waveform_to_points_varying_data_type(self):
        """
        Test to generate points from a waveform with varying data types for tensors raises no errors
        """
        configs = {}
        configs["plateau_length"] = 80
        configs["slope_length"] = 20
        tensor = torch.randn(1, 1)
        waveform_mgr = WaveformManager(configs)
        try:
            waveform = waveform_mgr.points_to_waveform(
                tensor.to(TorchUtils.get_device()))
            waveform_mgr.waveform_to_points(waveform)
        except Exception:
            self.fail("Exception raised")
        waveform = TorchUtils.format(data=waveform,
                                     device=TorchUtils.get_device(),
                                     data_type=torch.float64)
        try:
            waveform_mgr.waveform_to_points(waveform)
        except Exception:
            self.fail("Exception raised")
        waveform = TorchUtils.format(data=waveform,
                                     device=TorchUtils.get_device(),
                                     data_type=torch.float16)
        try:
            waveform_mgr.waveform_to_points(waveform)
        except Exception:
            self.fail("Exception raised")

    def test_waveform_to_points_negative_plateau_slope(self):
        """
        Test to generate points from a waveform with a negative plateau value raises
        mask Index Error
        """
        manager = WaveformManager({"plateau_length": -1, "slope_length": 2})
        data = torch.tensor([[0], [1], [1], [1], [5], [5], [5], [0]],
                            device=TorchUtils.get_device(),
                            dtype=torch.get_default_dtype())
        with self.assertRaises(IndexError):
            manager.waveform_to_points(data)
        """
        Test to generate points from a waveform with a negative slope value raises ValueError
        """
        manager = WaveformManager({"plateau_length": 1, "slope_length": -2})
        data = torch.tensor([[0], [1], [1], [1], [5], [5], [5], [0]],
                            device=TorchUtils.get_device(),
                            dtype=torch.get_default_dtype())
        with self.assertRaises(RuntimeError):
            manager.waveform_to_points(data)


if __name__ == "__main__":
    unittest.main()
