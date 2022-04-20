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
    Class for testing the method - points_to_plateaus() in waveform.py.
    """
    def test_points_to_plateaus(self):
        """
        Test to generate a plateau for the points inputted and checking with
        all values forming a plateau.
        """
        configs = {}
        configs["plateau_length"] = 80
        configs["slope_length"] = 20
        waveform_mgr = WaveformManager(configs)
        data = (1, 1)
        points = torch.rand(data)
        plateau = waveform_mgr.points_to_plateaus(points)
        point_value = points.tolist()[0]
        plateau_values = plateau.tolist()
        self.assertEqual(
            point_value,
            plateau_values[random.randint(0,
                                          len(plateau_values) - 1)])
        self.assertEqual(point_value, plateau_values[0])

    def test_points_to_plateaus_slope_plateau_0(self):
        """
        Test to generate a waveform with slope length = 0
        """
        configs = {}
        configs["plateau_length"] = 80
        configs["slope_length"] = 0
        waveform_mgr = WaveformManager(configs)
        points = torch.randint(3, 5, (1, ))
        plateau = waveform_mgr.points_to_plateaus(
            points.to(TorchUtils.get_device()))
        plateau_values = plateau.tolist()
        result = False
        if len(plateau_values) > 0:
            result = all(elem == plateau_values[0] for elem in plateau_values)
        self.assertEqual(result, True)
        self.assertEqual(len(plateau_values), 80)
        """
        Test to generate a plateau with plateau length = 0 and accurate shape of tensor
        """
        configs = {}
        configs["plateau_length"] = 0
        configs["slope_length"] = 20
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            waveform_mgr = WaveformManager(configs)
            self.assertEqual(len(caught_warnings), 1)
        points = torch.rand(2, 2)
        plateau = waveform_mgr.points_to_plateaus(
            points.to(TorchUtils.get_device()))
        plateau_values = plateau.tolist()
        self.assertEqual(plateau_values, [])
        """
        Test to generate a plateau with both slope and plateau length = 0
        """
        configs = {}
        configs["plateau_length"] = 0
        configs["slope_length"] = 0
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            waveform_mgr = WaveformManager(configs)
            self.assertEqual(len(caught_warnings), 2)
        points = torch.rand(2, 2)
        plateau = waveform_mgr.points_to_plateaus(
            points.to(TorchUtils.get_device()))
        plateau_values = plateau.tolist()
        self.assertEqual(plateau_values, [])

    def test_points_to_plateaus_negative(self):
        """
        Test to generate a plateau with negative values raises no errors
        """
        configs = {}
        configs["plateau_length"] = 80
        configs["slope_length"] = 20
        waveform_mgr = WaveformManager(configs)
        points = torch.tensor([[1., -1.], [1., -1.]])
        try:
            waveform_mgr.points_to_plateaus(points.to(TorchUtils.get_device()))
        except Exception:
            self.fail("Exception raised")

    def test_points_to_plateaus_nonetype_tensor(self):
        """
        Test to generate a plateau with tensor of a NoneType value which raises RuntimeError
        """
        configs = {}
        configs["plateau_length"] = 1
        configs["slope_length"] = 10
        waveform_mgr = WaveformManager(configs)
        with self.assertRaises(RuntimeError):
            waveform_mgr.points_to_plateaus(
                torch.tensor(None).to(TorchUtils.get_device()))

    def test_points_to_plateaus_empty_tensor(self):
        """
        Test to generate a plateau with an empty tensor which raises no errors
        """
        configs = {}
        configs["plateau_length"] = 80
        configs["slope_length"] = 20
        waveform_mgr = WaveformManager(configs)
        points = torch.empty(1, 1)
        try:
            waveform_mgr.points_to_plateaus(points.to(TorchUtils.get_device()))
        except Exception:
            self.fail("Exception raised")

    def test_points_to_plateaus_single_val(self):
        """
        Test to generate a plateau with a tensor containing a single value which raises no errors
        """
        configs = {}
        configs["plateau_length"] = 80
        configs["slope_length"] = 20
        waveform_mgr = WaveformManager(configs)
        points = torch.tensor([[1]])
        try:
            waveform_mgr.points_to_plateaus(points.to(TorchUtils.get_device()))
        except Exception:
            self.fail("Exception raised")

    def test_points_to_plateaus_invalid_type(self):
        """
        Test to generate a plateau with invalid type raises AttributeError
        """
        configs = {}
        configs["plateau_length"] = 1
        configs["slope_length"] = 10
        waveform_mgr = WaveformManager(configs)
        with self.assertRaises(AttributeError):
            waveform_mgr.points_to_plateaus([1, 2, 3, 4])
        with self.assertRaises(AttributeError):
            waveform_mgr.points_to_plateaus(np.array([1, 2, 3, 4]))
        with self.assertRaises(AttributeError):
            waveform_mgr.points_to_plateaus("String type is not accepted")
        with self.assertRaises(AttributeError):
            waveform_mgr.points_to_plateaus(None)

    def test_points_to_plateau_varying_data_type(self):
        """
        Test to generate a plateau with varying data types for tensors raises no errors
        """
        configs = {}
        configs["plateau_length"] = 1
        configs["slope_length"] = 10
        tensor = torch.randn(2, 2)
        waveform_mgr = WaveformManager(configs)
        try:
            waveform_mgr.points_to_plateaus(tensor.to(TorchUtils.get_device()))
        except Exception:
            self.fail("Exception raised")
        tensor = TorchUtils.format(data=tensor,
                                   device=TorchUtils.get_device(),
                                   data_type=torch.float64)
        try:
            waveform_mgr.points_to_plateaus(tensor)
        except Exception:
            self.fail("Exception raised")
        tensor = TorchUtils.format(data=tensor,
                                   device=TorchUtils.get_device(),
                                   data_type=torch.float16)
        try:
            waveform_mgr.points_to_plateaus(tensor)
        except Exception:
            self.fail("Exception raised")

    def test_points_to_plateaus_negative_plateau_slope(self):
        """
        Test to generate a plateau with a negative plateau value raises Runtime Error
        """
        configs = {}
        configs["plateau_length"] = -80
        configs["slope_length"] = 20
        waveform_mgr = WaveformManager(configs)
        data = (1, 1)
        points = torch.rand(data)
        with self.assertRaises(RuntimeError):
            waveform_mgr.points_to_plateaus(points.to(TorchUtils.get_device()))
        """
        Test to generate a plateau with a negative slope value raises no errors
        """
        configs = {}
        configs["plateau_length"] = 80
        configs["slope_length"] = -20
        waveform_mgr = WaveformManager(configs)
        data = (1, 1)
        points = torch.rand(data)
        try:
            waveform_mgr.points_to_plateaus(points.to(TorchUtils.get_device()))
        except Exception:
            self.fail("Exception raised")

    def test_points_to_plateaus_random(self):
        """
        Test to generate a plateau with random slope and plateau numbers
        raises no errors
        """
        configs = {}
        configs["plateau_length"] = random.randint(1, 100000)
        configs["slope_length"] = random.randint(1, 100000)
        waveform_mgr = WaveformManager(configs)
        points = torch.tensor([[1]])
        try:
            waveform_mgr.points_to_plateaus(points)
        except Exception:
            self.fail("Exception raised")


if __name__ == "__main__":
    unittest.main()
