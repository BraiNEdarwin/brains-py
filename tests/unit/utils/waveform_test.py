"""
Module for testing waveform transformations.
"""

import unittest as unittest
import torch
from brainspy.utils.waveform import WaveformManager
from brainspy.utils.pytorch import TorchUtils


class WaveformTest(unittest.TestCase):
    """
    Class for testing the waveform transformations. For each transform:
    -test fully on a small example
    -for randomly generated large examples:
        -does the result have the right size?
        -does the inverse transform bring us back to the original data?
        -do all waveforms start and end with 0?
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
                    device=TorchUtils.get_accelerator_type(),
                    dtype=TorchUtils.get_data_type(),
                ))

    def check_waveform_start_end(self, waveform):
        """
        Check if a waveform starts and ends with 0.
        """
        self.assertTrue((waveform[0, :] == 0.0).all(),
                        "Waveforms do not start with zero.")
        self.assertTrue((waveform[-1, :] == 0.0).all(),
                        "Waveforms do not end with zero.")

    def test_generate_mask_base(self):
        pass

    def test_expand(self):
        pass

    def test_points_to_waveform(self):
        pass

    def test_points_to_plateaus(self):
        pass

    def test_tile(self):
        pass

    def test_plateaus_to_waveform(self):
        """
        Test the transform from plateaus to waveform.
        """
        manager = WaveformManager({"plateau_length": 2, "slope_length": 2})
        data = torch.tensor([[1], [1], [3], [3]],
                            device=TorchUtils.get_accelerator_type(),
                            dtype=TorchUtils.get_data_type())
        output_data, output_mask = manager.plateaus_to_waveform(data)
        self.assertTrue(
            torch.equal(
                output_data,
                torch.tensor([[0.], [1.], [1.], [1.], [1.], [3.], [3.], [3.],
                              [3.], [0.]],
                             device=TorchUtils.get_accelerator_type(),
                             dtype=TorchUtils.get_data_type())),
            "Plateaus to waveform error")
        self.assertTrue(
            torch.equal(
                output_mask,
                torch.tensor([
                    False, False, True, True, False, False, True, True, False,
                    False
                ],
                             device=TorchUtils.get_accelerator_type(),
                             dtype=torch.bool)), "Plateaus to waveform error")
        for points in self.test_points:
            plateaus = self.waveform_mgr.points_to_plateaus(points)
            waveform, mask = self.waveform_mgr.plateaus_to_waveform(plateaus)
            self.check_waveform_start_end(waveform)
            self.assertEqual(
                len(waveform),
                len(plateaus) + self.waveform_mgr.slope_length *
                (len(points) + 1),
                "Plateaus to waveform - wrong length of result")
            plateaus_reverse = self.waveform_mgr.waveform_to_plateaus(waveform)
            self.assertTrue(torch.equal(plateaus, plateaus_reverse),
                            "Plateaus to waveform error")
            self.assertEqual(len(waveform), len(mask),
                             "Plateaus to waveform - wrong size of mask")
            self.assertEqual(len(waveform[mask]),
                             self.waveform_mgr.plateau_length * len(points),
                             "Plateaus to waveform mask error")

    def test_plateaus_to_points(self):
        """
        Test the transform from plateaus to points.
        """
        manager = WaveformManager({"plateau_length": 4, "slope_length": 2})
        data = torch.tensor(
            [[1], [1], [1], [1], [5], [5], [5], [5], [3], [3], [3], [3]],
            device=TorchUtils.get_accelerator_type(),
            dtype=TorchUtils.get_data_type())
        output = manager.plateaus_to_points(data)
        self.assertTrue(
            torch.equal(
                output,
                torch.tensor([[1], [5], [3]],
                             device=TorchUtils.get_accelerator_type(),
                             dtype=TorchUtils.get_data_type())),
            "Plateaus to points error")
        for points in self.test_points:
            plateaus = self.waveform_mgr.points_to_plateaus(points)
            points_reverse = self.waveform_mgr.plateaus_to_points(plateaus)
            self.assertTrue(torch.allclose(points, points_reverse),
                            "Plateaus to points error")

    def test_waveform_to_points(self):
        """
        Test the transform from waveform to points.
        """
        manager = WaveformManager({"plateau_length": 1, "slope_length": 2})
        data = torch.tensor([[0], [1], [1], [1], [5], [5], [5], [0]],
                            device=TorchUtils.get_accelerator_type(),
                            dtype=TorchUtils.get_data_type())
        output = manager.waveform_to_points(data)
        self.assertTrue(
            torch.equal(
                output,
                torch.tensor([[1], [5]],
                             device=TorchUtils.get_accelerator_type(),
                             dtype=TorchUtils.get_data_type())),
            "Plateaus to points error")
        for points in self.test_points:
            waveform = self.waveform_mgr.points_to_waveform(points)
            points_reverse = self.waveform_mgr.waveform_to_points(waveform)
            self.assertTrue(torch.allclose(points, points_reverse),
                            "Waveform to points error")

    def test_waveform_to_plateaus(self):
        """
        Test the transform from waveform to plateaus.
        """
        manager = WaveformManager({"plateau_length": 2, "slope_length": 2})
        data = torch.tensor([[0], [1], [1], [1], [1], [5], [5], [5], [5], [0]],
                            device=TorchUtils.get_accelerator_type(),
                            dtype=TorchUtils.get_data_type())
        output = manager.waveform_to_plateaus(data)
        self.assertTrue(
            torch.equal(
                output,
                torch.tensor([[1], [1], [5], [5]],
                             device=TorchUtils.get_accelerator_type(),
                             dtype=TorchUtils.get_data_type())))
        for points in self.test_points:
            waveform = self.waveform_mgr.points_to_waveform(points)
            plateaus = self.waveform_mgr.waveform_to_plateaus(waveform)
            self.assertEqual(len(plateaus),
                             self.waveform_mgr.plateau_length * len(points))
            waveform_reverse, _ = self.waveform_mgr.plateaus_to_waveform(
                plateaus)
            self.assertTrue(torch.equal(waveform, waveform_reverse),
                            "Plateaus to waveform error")

    def test_generate_mask(self):
        """
        Test generating a mask.
        """
        configs = {"plateau_length": 2, "slope_length": 1}
        manager = WaveformManager(configs)
        output = manager.generate_mask(7)
        self.assertTrue(
            torch.equal(
                output,
                torch.tensor([False, True, True, False, True, True, False],
                             device=TorchUtils.get_accelerator_type(),
                             dtype=torch.bool)), "Generate mask error")
        for points in self.test_points:
            waveform = self.waveform_mgr.points_to_waveform(points)
            mask = self.waveform_mgr.generate_mask(len(waveform))
            self.assertEqual(len(mask), len(waveform),
                             "Generate mask - wrong size.")
            self.assertEqual(len(waveform[mask]),
                             self.waveform_mgr.plateau_length * len(points),
                             "Generate mask error")


if __name__ == "__main__":
    unittest.main()
