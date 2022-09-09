"""
Module for testing waveform transformations.
"""
import unittest
import torch
import warnings
import random
import numpy as np
from brainspy.utils.waveform import WaveformManager
from brainspy.utils.pytorch import TorchUtils


class WaveformTest(unittest.TestCase):
    """
    Class for testing the method - waveform_to_plateaus() in waveform.py.
    """
    def test_waveform_to_plateaus_manual(self):
        """
        Test the transform from waveform to plateaus.
        """
        manager = WaveformManager({"plateau_length": 2, "slope_length": 2})
        data = torch.tensor([[0], [1], [1], [1], [1], [5], [5], [5], [5], [0]],
                            device=TorchUtils.get_device(),
                            dtype=torch.get_default_dtype())
        output = manager.waveform_to_plateaus(data)
        self.assertTrue(
            torch.equal(
                output,
                torch.tensor([[1], [1], [5], [5]],
                             device=TorchUtils.get_device(),
                             dtype=torch.get_default_dtype())))

    def test_waveform_to_plateaus_different_sizes(self):
        """
        Test to transform from waveform to plateaus with tensors of different sizes.
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
            plateaus = waveform_mgr.waveform_to_plateaus(waveform)
            self.assertEqual(len(plateaus),
                             waveform_mgr.plateau_length * len(points))
            waveform_reverse, _ = waveform_mgr.plateaus_to_waveform(plateaus)
            self.assertTrue(torch.equal(waveform, waveform_reverse),
                            "Plateaus to waveform error")

    def test_waveform_to_plateaus_slope_plateau_0(self):
        """
        Test to transform from waveform to plateaus with slope length = 0 raises mask IndexError
        """
        configs = {}
        configs["plateau_length"] = 80
        configs["slope_length"] = 0
        waveform_mgr = WaveformManager(configs)
        points = torch.randint(3, 5, (10, 1))
        waveform = waveform_mgr.points_to_waveform(
            points.to(TorchUtils.get_device()))
        with self.assertRaises(IndexError):
            waveform_mgr.waveform_to_plateaus(waveform)
        """
        Test to transform from waveform to plateaus with plateau length = 0 raises no errors
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
        try:
            plateaus = waveform_mgr.waveform_to_plateaus(waveform)
        except (Exception):
            self.fail("Exception raised")
        self.assertEqual(len(plateaus),
                         waveform_mgr.plateau_length * len(points))
        """
        Test to transform from waveform to plateaus with both slope and plateau length = 0
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
            waveform_mgr.waveform_to_plateaus(waveform)

    def test_waveform_to_plateaus_negative(self):
        """
        Test to transform from waveform to plateaus with negative values raises no errors
        """
        configs = {}
        configs["plateau_length"] = 80
        configs["slope_length"] = 20
        waveform_mgr = WaveformManager(configs)
        points = torch.tensor([[1., -1.], [1., -1.]])
        try:
            waveform = waveform_mgr.points_to_waveform(
                points.to(TorchUtils.get_device()))
            waveform_mgr.waveform_to_plateaus(waveform)
        except Exception:
            self.fail("Exception raised")

    def test_waveform_to_plateaus_nonetype_tensor(self):
        """
        Test to transform from waveform to plateaus with tensor of a NoneType value which raises
        RuntimeError
        """
        configs = {}
        configs["plateau_length"] = 1
        configs["slope_length"] = 10
        waveform_mgr = WaveformManager(configs)
        with self.assertRaises(RuntimeError):
            waveform_mgr.waveform_to_plateaus(
                torch.tensor(None).to(TorchUtils.get_device()))

    def test_waveform_to_plateaus_empty_tensor(self):
        """
        Test to transform from waveform to plateaus with an empty tensor which raises no errors
        """
        configs = {}
        configs["plateau_length"] = 80
        configs["slope_length"] = 20
        waveform_mgr = WaveformManager(configs)
        points = torch.empty(1, 1)
        try:
            waveform = waveform_mgr.points_to_waveform(
                points.to(TorchUtils.get_device()))
            waveform_mgr.waveform_to_plateaus(waveform)
        except Exception:
            self.fail("Exception raised")

    def test_waveform_to_plateaus_single_val(self):
        """
        Test to transform from waveform to plateaus with a tensor containing a single value which
        raises no errors
        """
        configs = {}
        configs["plateau_length"] = 80
        configs["slope_length"] = 20
        waveform_mgr = WaveformManager(configs)
        points = torch.tensor([[1]])
        try:
            waveform = waveform_mgr.points_to_waveform(
                points.to(TorchUtils.get_device()))
            waveform_mgr.waveform_to_plateaus(waveform)
        except Exception:
            self.fail("Exception raised")

    def test_waveform_to_plateaus_invalid_type(self):
        """
        Test to transform from waveform to plateaus with invalid type raises TypeError
        """
        configs = {}
        configs["plateau_length"] = 1
        configs["slope_length"] = 10
        waveform_mgr = WaveformManager(configs)
        with self.assertRaises(AssertionError):
            waveform_mgr.waveform_to_plateaus([1, 2, 3, 4])
        with self.assertRaises(AssertionError):
            waveform_mgr.waveform_to_plateaus(np.array(1))
        with self.assertRaises(AssertionError):
            waveform_mgr.waveform_to_plateaus("String type is not accepted")
        with self.assertRaises(AssertionError):
            waveform_mgr.waveform_to_plateaus(None)

    def test_waveform_to_plateaus_varying_data_type(self):
        """
        Test to transform from waveform to plateaus with varying data types for tensors raises
        no errors
        """
        configs = {}
        configs["plateau_length"] = 1
        configs["slope_length"] = 10
        tensor = torch.randn(2, 2)
        waveform_mgr = WaveformManager(configs)
        try:
            waveform = waveform_mgr.points_to_waveform(
                tensor.to(TorchUtils.get_device()))
            waveform_mgr.waveform_to_plateaus(waveform)
        except Exception:
            self.fail("Exception raised")
        waveform = TorchUtils.format(data=waveform,
                                     device=TorchUtils.get_device(),
                                     data_type=torch.float64)
        try:
            waveform_mgr.waveform_to_plateaus(waveform)
        except Exception:
            self.fail("Exception raised")
        waveform = TorchUtils.format(data=waveform,
                                     device=TorchUtils.get_device(),
                                     data_type=torch.float16)
        try:
            waveform_mgr.waveform_to_plateaus(waveform)
        except Exception:
            self.fail("Exception raised")

    def test_waveform_to_plateaus_negative_plateau_slope(self):
        """
        Test to transform from waveform to plateaus with a negative plateau value raises
        ZeroDivisionError Error
        """
        configs = {}
        configs["plateau_length"] = -2
        configs["slope_length"] = 2
        waveform_mgr = WaveformManager(configs)
        data = torch.tensor([[0], [1], [1], [1], [1], [5], [5], [5], [5], [0]],
                            device=TorchUtils.get_device(),
                            dtype=torch.get_default_dtype())
        with self.assertRaises(ZeroDivisionError):
            waveform_mgr.waveform_to_plateaus(data)
        """
        Test to transform from waveform to plateaus with a negative slope value raises ValueError
        """
        configs = {}
        configs["plateau_length"] = 2
        configs["slope_length"] = -2
        waveform_mgr = WaveformManager(configs)
        data = torch.tensor([[0], [1], [1], [1], [1], [5], [5], [5], [5], [0]],
                            device=TorchUtils.get_device(),
                            dtype=torch.get_default_dtype())
        with self.assertRaises(ZeroDivisionError):
            waveform_mgr.waveform_to_plateaus(data)

    def test_waveform_to_plateaus_random(self):
        """
        Test to transform from waveform to plateaus with random slope and plateau numbers
        raises no errors
        """
        configs = {}
        configs["plateau_length"] = random.randint(1, 100000)
        configs["slope_length"] = random.randint(1, 100000)
        waveform_mgr = WaveformManager(configs)
        points = torch.tensor([[1]])
        try:
            waveform = waveform_mgr.points_to_waveform(points)
            waveform_mgr.waveform_to_plateaus(waveform)
        except Exception:
            self.fail("Exception raised")


if __name__ == "__main__":
    unittest.main()
