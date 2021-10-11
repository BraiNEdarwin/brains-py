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
    Class for testing the method - points_to_waveform() in waveform.py.
    """
    def test_points_to_waveform_values(self):
        """
        Test to generate a waveform and
        checking first, final and middle values.
        """
        configs = {}
        configs["plateau_length"] = 80
        configs["slope_length"] = 20
        waveform_mgr = WaveformManager(configs)
        data = (1, 1)
        points = torch.rand(data)
        waveform = waveform_mgr.points_to_waveform(
            points.to(TorchUtils.get_device()))
        point_value = points.tolist()[0]
        waveform_values = waveform.tolist()
        max_wave = max(waveform_values)
        self.assertEqual(max_wave, point_value)
        self.assertEqual(waveform_values[0], [0.0])
        self.assertEqual(waveform_values[len(waveform_values) - 1], [0.0])

    def test_points_to_waveform_slope_plateau_0(self):
        """
        Test to generate a waveform with slope length = 0
        """
        configs = {}
        configs["plateau_length"] = 80
        configs["slope_length"] = 0
        waveform_mgr = WaveformManager(configs)
        points = torch.randint(3, 5, (1, ))
        waveform = waveform_mgr.points_to_waveform(
            points.to(TorchUtils.get_device()))
        waveform_values = waveform.tolist()
        result = False
        if len(waveform_values) > 0:
            result = all(elem == waveform_values[0]
                         for elem in waveform_values)
        self.assertEqual(result, True)
        self.assertEqual(len(waveform_values), 80)
        """
        Test to generate waveform with plateau length = 0 and accurate shape of tensor
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
        waveform_values = waveform.tolist()
        for i in range(1, len(waveform_values) - 2):
            self.assertTrue(waveform_values[i] != [0.0, 0.0])
        self.assertEqual(waveform_values[0], [0.0, 0.0])
        self.assertEqual(waveform_values[len(waveform_values) - 1], [0.0, 0.0])
        """
        Test to generate waveform with both slope and plateau length = 0
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
        waveform_values = waveform.tolist()
        self.assertEqual(waveform_values, [])

    def test_points_to_waveform_negative(self):
        """
        Test to generate a waveform with negative values raises no errors
        """
        configs = {}
        configs["plateau_length"] = 80
        configs["slope_length"] = 20
        waveform_mgr = WaveformManager(configs)
        points = torch.tensor([[1., -1.], [1., -1.]])
        try:
            waveform_mgr.points_to_waveform(points.to(TorchUtils.get_device()))
        except Exception:
            self.fail("Exception raised")

    def test_points_to_waveform_nonetype_tensor(self):
        """
        Test to generate a waveform with tensor of a NoneType value which raises RuntimeError
        """
        configs = {}
        configs["plateau_length"] = 1
        configs["slope_length"] = 10
        waveform_mgr = WaveformManager(configs)
        with self.assertRaises(RuntimeError):
            waveform_mgr.points_to_waveform(
                torch.tensor(None).to(TorchUtils.get_device()))

    def test_points_to_waveform_empty_tensor(self):
        """
        Test to generate a waveform with an empty tensor which raises no errors
        """
        configs = {}
        configs["plateau_length"] = 80
        configs["slope_length"] = 20
        waveform_mgr = WaveformManager(configs)
        points = torch.empty(1, 1)
        try:
            waveform_mgr.points_to_waveform(points.to(TorchUtils.get_device()))
        except Exception:
            self.fail("Exception raised")

    def test_points_to_waveform_single_val(self):
        """
        Test to generate a waveform with a tensor containing a single value which raises no errors
        """
        configs = {}
        configs["plateau_length"] = 80
        configs["slope_length"] = 20
        waveform_mgr = WaveformManager(configs)
        points = torch.tensor([[1]])
        try:
            waveform_mgr.points_to_waveform(points.to(TorchUtils.get_device()))
        except Exception:
            self.fail("Exception raised")

    def test_points_to_waveform_fail(self):
        """
        Runtime error raised when plateau length is 0 and tensor dimension does not match
        """
        configs = {}
        configs["plateau_length"] = 0
        configs["slope_length"] = 20
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            waveform_mgr = WaveformManager(configs)
            self.assertEqual(len(caught_warnings), 1)
        points = torch.rand(1)
        with self.assertRaises(RuntimeError):
            waveform_mgr.points_to_waveform(points.to(TorchUtils.get_device()))

    def test_points_to_waveform_invalid_type(self):
        """
        Test to generate a waveform with invalid type raises AttributeError or TypeError
        """
        configs = {}
        configs["plateau_length"] = 1
        configs["slope_length"] = 10
        waveform_mgr = WaveformManager(configs)
        with self.assertRaises(AttributeError):
            waveform_mgr.points_to_waveform([1, 2, 3, 4])
        with self.assertRaises(AttributeError):
            waveform_mgr.points_to_waveform(np.array([1, 2, 3, 4]))
        with self.assertRaises(AttributeError):
            waveform_mgr.points_to_waveform("String type is not accepted")
        with self.assertRaises(TypeError):
            waveform_mgr.points_to_waveform(None)

    def test_points_to_waveform_varying_data_type(self):
        """
        Test to generate a waveform with varying data types for tensors raises no errors
        """
        configs = {}
        configs["plateau_length"] = 1
        configs["slope_length"] = 10
        tensor = torch.randn(2, 2)
        waveform_mgr = WaveformManager(configs)
        try:
            waveform_mgr.points_to_waveform(tensor.to(TorchUtils.get_device()))
        except Exception:
            self.fail("Exception raised")
        tensor = TorchUtils.format(data=tensor,
                                   device=TorchUtils.get_device(),
                                   data_type=torch.float64)
        try:
            waveform_mgr.points_to_waveform(tensor)
        except Exception:
            self.fail("Exception raised")
        tensor = TorchUtils.format(data=tensor,
                                   device=TorchUtils.get_device(),
                                   data_type=torch.float16)
        try:
            waveform_mgr.points_to_waveform(tensor)
        except Exception:
            self.fail("Exception raised")

    def test_points_to_waveform_negative_plateau_slope(self):
        """
        Test to generate a waveform with a negative plateau value raises Runtime Error
        """
        configs = {}
        configs["plateau_length"] = -80
        configs["slope_length"] = 20
        waveform_mgr = WaveformManager(configs)
        data = (1, 1)
        points = torch.rand(data)
        with self.assertRaises(RuntimeError):
            waveform_mgr.points_to_waveform(points.to(TorchUtils.get_device()))
        """
        Test to generate a waveform with a negative slope value raises ValueError
        """
        configs = {}
        configs["plateau_length"] = 80
        configs["slope_length"] = -20
        waveform_mgr = WaveformManager(configs)
        data = (1, 1)
        points = torch.rand(data)
        with self.assertRaises(ValueError):
            waveform_mgr.points_to_waveform(points.to(TorchUtils.get_device()))

    def test_points_to_waveform_random(self):
        """
        Test to generate a waveform with random slope and plateau numbers
        raises no errors
        """
        configs = {}
        configs["plateau_length"] = random.randint(1, 100000)
        configs["slope_length"] = random.randint(1, 100000)
        waveform_mgr = WaveformManager(configs)
        points = torch.tensor([[1]])
        try:
            waveform_mgr.points_to_waveform(points)
        except Exception:
            self.fail("Exception raised")


if __name__ == "__main__":
    unittest.main()
