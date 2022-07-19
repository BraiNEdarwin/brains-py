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
    Class for testing the method - plateaus_to_waveform() in waveform.py.
    """
    def test_plateaus_to_waveform_manual_numpy(self):
        """
        Test to transform from plateaus to waveform and checking the output values and mask.
        """
        configs = {"plateau_length": 2, "slope_length": 2}
        manager = WaveformManager(configs)
        data = torch.tensor([[1], [1], [3], [3]],
                            device=TorchUtils.get_device(),
                            dtype=torch.get_default_dtype())
        output_data, output_mask = manager.plateaus_to_waveform(data, False)
        self.assertTrue(
            np.equal(
                output_data,
                np.array([[0.0], [0.5], [1.0], [1.0], [1.6666667461395264],
                          [2.3333334922790527], [3.0], [3.0], [1.5],
                          [0.0]])).all(), "Plateaus to waveform error")
        self.assertTrue(
            np.equal(
                output_mask,
                np.array([
                    False, False, True, True, False, False, True, True, False,
                    False
                ])).all(), "Plateaus to waveform error")

    def test_plateaus_to_waveform_manual(self):
        """
        Test to transform from plateaus to waveform and checking the output values and mask.
        """
        configs = {"plateau_length": 2, "slope_length": 2}
        manager = WaveformManager(configs)
        data = torch.tensor([[1], [1], [3], [3]],
                            device=TorchUtils.get_device(),
                            dtype=torch.get_default_dtype())
        output_data, output_mask = manager.plateaus_to_waveform(data)
        self.assertTrue(
            torch.equal(
                output_data,
                torch.tensor(
                    [[0.0], [0.5], [1.0], [1.0], [1.6666667461395264],
                     [2.3333334922790527], [3.0], [3.0], [1.5], [0.0]],
                    device=TorchUtils.get_device(),
                    dtype=torch.get_default_dtype())),
            "Plateaus to waveform error")
        self.assertTrue(
            torch.equal(
                output_mask,
                torch.tensor([
                    False, False, True, True, False, False, True, True, False,
                    False
                ],
                             device=TorchUtils.get_device(),
                             dtype=torch.bool)), "Plateaus to waveform error")

    def test_plateaus_to_waveform_differnt_sizes(self):
        """
        Test to generate a waveform from a plateau by checking with various sizes
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
            waveform, mask = waveform_mgr.plateaus_to_waveform(plateaus)
            self.check_waveform_start_end(waveform)
            self.assertEqual(
                len(waveform),
                len(plateaus) + waveform_mgr.slope_length * (len(points) + 1),
                "Plateaus to waveform - wrong length of result")
            plateaus_reverse = waveform_mgr.waveform_to_plateaus(waveform)
            self.assertTrue(torch.equal(plateaus, plateaus_reverse),
                            "Plateaus to waveform error")
            self.assertEqual(len(waveform), len(mask),
                             "Plateaus to waveform - wrong size of mask")
            self.assertEqual(len(waveform[mask]),
                             waveform_mgr.plateau_length * len(points),
                             "Plateaus to waveform mask error")

    def check_waveform_start_end(self, waveform):
        """
        Check if a waveform starts and ends with 0.
        This is a helper function for the above test.
        """
        self.assertTrue((waveform[0, :] == 0.0).all(),
                        "Waveforms do not start with zero.")
        self.assertTrue((waveform[-1, :] == 0.0).all(),
                        "Waveforms do not end with zero.")

    def test_plateaus_to_waveform_random_points(self):
        """
        Test to generate a plateau from random points and subsequently
        generate a waveform with that plateau.
        """
        configs = {}
        configs["plateau_length"] = 80
        configs["slope_length"] = 20
        waveform_mgr = WaveformManager(configs)
        data = (1, 1)
        points = torch.rand(data)
        plateau = waveform_mgr.points_to_plateaus(points)
        waveform, mask = waveform_mgr.plateaus_to_waveform(plateau)
        point_value = points.tolist()[0]
        waveform_values = waveform.tolist()
        mask_values = mask.tolist()
        max_wave = max(waveform_values)
        self.assertEqual(max_wave, point_value)
        self.assertEqual(waveform_values[0], [0.0])
        self.assertEqual(waveform_values[len(waveform_values) - 1], [0.0])
        self.assertEqual(waveform_values[len(mask_values) - 1], [0.0])

    def test_plateaus_to_waveform_slope_plateau_0(self):
        """
        Test to generate a waveform from a plateau with slope length = 0
        """
        configs = {}
        configs["plateau_length"] = 80
        configs["slope_length"] = 0
        waveform_mgr = WaveformManager(configs)
        points = torch.randint(3, 5, (1, ))
        plateau = waveform_mgr.points_to_plateaus(
            points.to(TorchUtils.get_device()))
        waveform, mask = waveform_mgr.plateaus_to_waveform(plateau)
        waveform_values = waveform.tolist()
        mask_values = mask.tolist()
        result = False
        if len(waveform_values) > 0:
            result = all(elem == waveform_values[0]
                         for elem in waveform_values)
        self.assertEqual(result, True)
        self.assertEqual(len(waveform_values), 80)
        self.assertEqual(len(mask_values), 80)
        """
        Test to generate a waveform with plateau length = 0 raises ZeroDivisionError
        """
        configs = {}
        configs["plateau_length"] = 0
        configs["slope_length"] = 20
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            waveform_mgr = WaveformManager(configs)
            self.assertEqual(len(caught_warnings), 1)
        plateau = torch.tensor(
            [[0.0], [0.5], [1.0], [1.0], [1.6666667461395264],
             [2.3333334922790527], [3.0], [3.0], [1.5], [0.0]],
            device=TorchUtils.get_device(),
            dtype=torch.get_default_dtype())
        with self.assertRaises(ZeroDivisionError):
            waveform_mgr.plateaus_to_waveform(plateau)
        """
        Test to generate a waveform with both slope and plateau length = 0 raises ZeroDivisonError
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
        with self.assertRaises(ZeroDivisionError):
            waveform_mgr.plateaus_to_waveform(plateau)

    def test_plateaus_to_waveform_negative(self):
        """
        Test to generate a waveform from plateaus with negative
        values raises no errors
        """
        configs = {}
        configs["plateau_length"] = 80
        configs["slope_length"] = 20
        waveform_mgr = WaveformManager(configs)
        points = torch.tensor([[1., -1.], [1., -1.]])
        try:
            plateau = waveform_mgr.points_to_plateaus(
                points.to(TorchUtils.get_device()))
            waveform_mgr.plateaus_to_waveform(plateau)
        except Exception:
            self.fail("Exception raised")

    def test_plateaus_to_waveform_nonetype_tensor(self):
        """
        Test to generate a waveform with tensor of a NoneType value which raises RuntimeError
        """
        configs = {}
        configs["plateau_length"] = 1
        configs["slope_length"] = 10
        waveform_mgr = WaveformManager(configs)
        with self.assertRaises(RuntimeError):
            waveform_mgr.plateaus_to_waveform(
                torch.tensor(None).to(TorchUtils.get_device()))

    def test_plateaus_to_waveform_empty_tensor(self):
        """
        Test to generate a waveform with a plateau as an empty tensor which raises no errors
        """
        configs = {}
        configs["plateau_length"] = 80
        configs["slope_length"] = 20
        waveform_mgr = WaveformManager(configs)
        points = torch.empty(1, 1)
        try:
            plateau = waveform_mgr.points_to_plateaus(
                points.to(TorchUtils.get_device()))
            waveform_mgr.plateaus_to_waveform(plateau)
        except Exception:
            self.fail("Exception raised")

    def test_plateaus_to_waveform_single_val(self):
        """
        Test to generate a waveform from a plateau containing a single value which raises no errors
        """
        configs = {}
        configs["plateau_length"] = 80
        configs["slope_length"] = 20
        waveform_mgr = WaveformManager(configs)
        points = torch.tensor([[1]])
        try:
            plateau = waveform_mgr.points_to_plateaus(
                points.to(TorchUtils.get_device()))
            waveform_mgr.plateaus_to_waveform(plateau)
        except Exception:
            self.fail("Exception raised")

    def test_plateaus_to_waveform_invalid_type(self):
        """
        Test to generate a waveform from a plateau with invalid type raises AttributeError
        or TypeError
        """
        configs = {}
        configs["plateau_length"] = 1
        configs["slope_length"] = 10
        waveform_mgr = WaveformManager(configs)
        with self.assertRaises(AttributeError):
            waveform_mgr.plateaus_to_waveform([1, 2, 3, 4])
        with self.assertRaises(AttributeError):
            waveform_mgr.plateaus_to_waveform(np.array([1, 2, 3, 4]))
        with self.assertRaises(AttributeError):
            waveform_mgr.plateaus_to_waveform("String type is not accepted")
        with self.assertRaises(TypeError):
            waveform_mgr.plateaus_to_waveform(None)

    def test_plateaus_to_waveform_varying_data_type(self):
        """
        Test to generate a waveform with a plateau of varying data types raises no errors
        """
        configs = {}
        configs["plateau_length"] = 1
        configs["slope_length"] = 10
        tensor = torch.randn(2, 2)
        waveform_mgr = WaveformManager(configs)
        plateau = waveform_mgr.points_to_plateaus(
            tensor.to(TorchUtils.get_device()))
        try:
            waveform_mgr.plateaus_to_waveform(plateau)
        except Exception:
            self.fail("Exception raised")
        plateau = TorchUtils.format(data=plateau,
                                    device=TorchUtils.get_device(),
                                    data_type=torch.float64)
        try:
            waveform_mgr.plateaus_to_waveform(plateau)
        except Exception:
            self.fail("Exception raised")
        plateau = TorchUtils.format(data=plateau,
                                    device=TorchUtils.get_device(),
                                    data_type=torch.float16)
        try:
            waveform_mgr.plateaus_to_waveform(plateau)
        except Exception:
            self.fail("Exception raised")

    def test_plateaus_to_waveform_negative_plateau_slope(self):
        """
        Test to generate a waveform from a plateau, with a negative plateau value raises no errors
        """
        configs = {}
        configs["plateau_length"] = -10
        configs["slope_length"] = 20
        waveform_mgr = WaveformManager(configs)
        plateau = torch.tensor(
            [[0.0], [0.5], [1.0], [1.0], [1.6666667461395264],
             [2.3333334922790527], [3.0], [3.0], [1.5], [0.0]],
            device=TorchUtils.get_device(),
            dtype=torch.get_default_dtype())
        try:
            waveform_mgr.plateaus_to_waveform(plateau)
        except Exception:
            self.fail("Exception raised")
        """
        Test to generate a plateau from a waveform, with a negative slope value raises ValueError
        """
        configs = {}
        configs["plateau_length"] = 10
        configs["slope_length"] = -20
        waveform_mgr = WaveformManager(configs)
        plateau = torch.tensor(
            [[0.0], [0.5], [1.0], [1.0], [1.6666667461395264],
             [2.3333334922790527], [3.0], [3.0], [1.5], [0.0]],
            device=TorchUtils.get_device(),
            dtype=torch.get_default_dtype())
        with self.assertRaises(ValueError):
            waveform_mgr.plateaus_to_waveform(plateau)

    def test_plateaus_to_wavefrom_random(self):
        """
        Test to generate a waveform from a plateau with random slope and plateau numbers
        raises no errors
        """
        configs = {}
        configs["plateau_length"] = random.randint(1, 100000)
        configs["slope_length"] = random.randint(1, 100000)
        waveform_mgr = WaveformManager(configs)
        points = torch.tensor([[1]])
        try:
            plateau = waveform_mgr.points_to_plateaus(points)
            waveform_mgr.plateaus_to_waveform(plateau)
        except Exception:
            self.fail("Exception raised")


if __name__ == "__main__":
    unittest.main()
