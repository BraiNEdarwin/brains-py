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
    Class for testing the method - plateaus_to_points() in waveform.py.
    """
    def test_plateaus_to_points_manual(self):
        """
        Test the transform from plateaus to points.
        """
        manager = WaveformManager({"plateau_length": 4, "slope_length": 2})
        data = torch.tensor(
            [[1], [1], [1], [1], [5], [5], [5], [5], [3], [3], [3], [3]],
            device=TorchUtils.get_device(),
            dtype=torch.get_default_dtype())
        output = manager.plateaus_to_points(data)
        self.assertTrue(
            torch.equal(
                output,
                torch.tensor([[1], [5], [3]],
                             device=TorchUtils.get_device(),
                             dtype=torch.get_default_dtype())),
            "Plateaus to points error")

    def test_plateaus_to_points_different_sizes(self):
        """
        Test to generate points from a plateau by checking with various sizes
        of tensors.
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
            plateaus = waveform_mgr.points_to_plateaus(points)
            points_reverse = waveform_mgr.plateaus_to_points(plateaus)
            self.assertTrue(torch.allclose(points, points_reverse),
                            "Plateaus to points error")

    def test_plateaus_to_points_slope_plateau_0(self):
        """
        Test to generate a points from a plateau with slope length = 0
        """
        configs = {}
        configs["plateau_length"] = 80
        configs["slope_length"] = 0
        waveform_mgr = WaveformManager(configs)
        points = torch.randint(3, 5, (1, ))
        plateau = waveform_mgr.points_to_plateaus(
            points.to(TorchUtils.get_device()))
        plateau_vals = []
        for i in plateau.tolist():
            intlist = []
            intlist.append(i)
            plateau_vals.append(intlist)
        output = waveform_mgr.plateaus_to_points(
            TorchUtils.format(plateau_vals))
        output_list = output.tolist()
        points_list = [item for sublist in output_list for item in sublist]
        self.assertEqual(points_list, points.tolist())
        """
        Test to generate points from a plateau with plateau length = 0
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
        plateau_vals = []
        for i in plateau.tolist():
            intlist = []
            intlist.append(i)
            plateau_vals.append(intlist)
        with self.assertRaises(ZeroDivisionError):
            waveform_mgr.plateaus_to_points(TorchUtils.format(plateau_vals))
        """
        Test to generate points from a plateau with both slope and plateau length = 0
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
        plateau_vals = []
        for i in plateau.tolist():
            intlist = []
            intlist.append(i)
            plateau_vals.append(intlist)
        with self.assertRaises(ZeroDivisionError):
            waveform_mgr.plateaus_to_points(TorchUtils.format(plateau_vals))

    def test_plateaus_to_points_negative(self):
        """
        Test to generate points from a plateau with negative values raises no errors
        """
        configs = {}
        configs["plateau_length"] = 4
        configs["slope_length"] = 2
        waveform_mgr = WaveformManager(configs)
        plateau = torch.tensor([[-1], [-1], [-1], [-1], [-5], [-5], [-5], [-5],
                                [-3], [-3], [-3], [-3]],
                               device=TorchUtils.get_device(),
                               dtype=torch.get_default_dtype())
        try:
            waveform_mgr.plateaus_to_points(plateau)
        except Exception:
            self.fail("Exception raised")

    def test_plateaus_to_points_nonetype_tensor(self):
        """
        Test to generate points from a plateau with tensor of a NoneType value which raises
        RuntimeError
        """
        configs = {}
        configs["plateau_length"] = 1
        configs["slope_length"] = 10
        waveform_mgr = WaveformManager(configs)
        with self.assertRaises(RuntimeError):
            waveform_mgr.plateaus_to_points(
                torch.tensor(None).to(TorchUtils.get_device()))

    def test_points_to_plateaus_invalid_type(self):
        """
        Test to generate points from a plateau with invalid type raises AttributeError
        """
        configs = {}
        configs["plateau_length"] = 1
        configs["slope_length"] = 10
        waveform_mgr = WaveformManager(configs)
        with self.assertRaises(AttributeError):
            waveform_mgr.plateaus_to_points([1, 2, 3, 4])
        with self.assertRaises(TypeError):
            waveform_mgr.plateaus_to_points(np.array(1))
        with self.assertRaises(AttributeError):
            waveform_mgr.plateaus_to_points("String type is not accepted")
        with self.assertRaises(TypeError):
            waveform_mgr.plateaus_to_points(None)

    def test_plateaus_to_points_varying_data_type(self):
        """
        Test to generate points from a plateau with varying data types for tensors raises no errors
        """
        configs = {}
        configs["plateau_length"] = 1
        configs["slope_length"] = 10
        tensor = torch.tensor(
            [[1], [1], [1], [1], [5], [5], [5], [5], [3], [3], [3], [3]],
            device=TorchUtils.get_device(),
            dtype=torch.get_default_dtype())
        waveform_mgr = WaveformManager(configs)
        try:
            waveform_mgr.plateaus_to_points(tensor.to(TorchUtils.get_device()))
        except Exception:
            self.fail("Exception raised")
        tensor = TorchUtils.format(data=tensor,
                                   device=TorchUtils.get_device(),
                                   data_type=torch.float64)
        try:
            waveform_mgr.plateaus_to_points(tensor)
        except Exception:
            self.fail("Exception raised")
        tensor = TorchUtils.format(data=tensor,
                                   device=TorchUtils.get_device(),
                                   data_type=torch.float16)
        try:
            waveform_mgr.plateaus_to_points(tensor)
        except Exception:
            self.fail("Exception raised")

    def test_plateaus_to_points_negative_plateau_slope(self):
        """
        Test to generate points from a plateau with a negative slope value raises no errors
        """
        configs = {}
        configs["plateau_length"] = 80
        configs["slope_length"] = -20
        waveform_mgr = WaveformManager(configs)
        points = torch.randint(3, 5, (1, ))
        plateau = waveform_mgr.points_to_plateaus(
            points.to(TorchUtils.get_device()))
        plateau_vals = []
        for i in plateau.tolist():
            intlist = []
            intlist.append(i)
            plateau_vals.append(intlist)
        try:
            waveform_mgr.plateaus_to_points(TorchUtils.format(plateau_vals))
        except (Exception):
            self.fail("Exception Raised")
        """
        Test to generate points from a plateau with a negative plateau value raises RuntimeError
        """
        configs = {}
        configs["plateau_length"] = -4
        configs["slope_length"] = 2
        waveform_mgr = WaveformManager(configs)
        plateau = torch.tensor(
            [[1], [1], [1], [1], [5], [5], [5], [5], [3], [3], [3], [3]],
            device=TorchUtils.get_device(),
            dtype=torch.get_default_dtype())
        with self.assertRaises(RuntimeError):
            waveform_mgr.plateaus_to_points(TorchUtils.format(plateau))


if __name__ == "__main__":
    unittest.main()
