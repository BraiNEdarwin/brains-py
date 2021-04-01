"""
Unit tests for the waveform manager
@author: ualegre
"""
import unittest2 as unittest
import torch
import random
from brainspy.utils.waveform import WaveformManager
from brainspy.utils.pytorch import TorchUtils


class WaveformTest(unittest.TestCase):
    def __init__(self, test_name):
        super(WaveformTest, self).__init__()
        configs = {}
        configs["plateau_length"] = 80
        configs["slope_length"] = 20
        self.configs = configs
        self.waveform_mgr = WaveformManager(configs)

    def full_check(self, point_no):
        points = torch.rand(
            point_no, device=TorchUtils.get_device(), dtype=torch.get_default_dtype()
        )  # .unsqueeze(dim=1)
        waveform = self.waveform_mgr.points_to_waveform(points)
        assert (waveform[0, :] == 0.0).all() and (
            waveform[-1, :] == 0.0
        ).all(), "Waveforms do not start and end with zero"
        assert len(waveform) == (
            (self.waveform_mgr.plateau_length * len(points))
            + (self.waveform_mgr.slope_length * (len(points) + 1))
        ), "Waveform has an incorrect shape"

        mask = self.waveform_mgr.generate_mask(len(waveform))
        assert len(mask) == len(waveform)

        waveform_to_points = self.waveform_mgr.waveform_to_points(waveform)
        plateaus_to_points = self.waveform_mgr.plateaus_to_points(waveform[mask])
        assert (
            (points.half().float() == waveform_to_points.half().float()).all()
            == (points.half().float() == plateaus_to_points.half().float()).all()
            == True
        ), "Inconsistent to_point conversion"

        points_to_plateau = self.waveform_mgr.points_to_plateaus(points)
        waveform_to_plateau = self.waveform_mgr.waveform_to_plateaus(waveform)
        assert (waveform[mask] == points_to_plateau).all() == (
            waveform[mask] == waveform_to_plateau
        ).all(), "Inconsistent plateau conversion"

        plateaus_to_waveform, _ = self.waveform_mgr.plateaus_to_waveform(waveform[mask])
        assert (
            waveform == plateaus_to_waveform
        ).all(), "Inconsistent waveform conversion"

    def test_init(self):
        """
        Test to check initialization of variables from configs
        """
        waveform = WaveformManager(self.configs)
        self.assertEqual(waveform.slope_length, 20)
        self.assertEqual(waveform.plateau_length, 80)

    def test_generate_mask_base(self):
        """
        Test to generate an initial and final mask for the torch tensor based on the configs of the waveform
        """
        waveform = WaveformManager(self.configs)
        waveform.generate_mask_base()
        final_mask_list = waveform.final_mask.tolist()
        initial_mask_list = waveform.initial_mask.tolist()
        self.assertEqual(initial_mask_list, ([False] * 20) + ([True] * 80))
        self.assertEqual(final_mask_list, [False] * 20)

    def test_expand(self):
        """
        Test to format the amplitudes and slopes to have the same length
        """
        waveform = WaveformManager(self.configs)
        plateau_lengths = waveform._expand(waveform.plateau_length, 100)
        self.assertEqual(len(plateau_lengths), 100)

    def test_points_to_waveform(self):
        """
        Test to generates a waveform with constant intervals of value and checking first,final and middle values
        """
        waveform_mgr = WaveformManager(self.configs)
        data = (1, 1)
        points = torch.rand(data)
        waveform = waveform_mgr.points_to_waveform(points)
        point_value = points.tolist()[0]
        waveform_values = waveform.tolist()
        max_wave = max(waveform_values)
        self.assertEqual(max_wave, point_value)
        self.assertEqual(waveform_values[0], [0.0])
        self.assertEqual(waveform_values[len(waveform_values) - 1], [0.0])

    def test_points_to_plateaus(self):
        """
        Test to generate plateaus for the points inputted and checking with all values forming a plateau
        """
        waveform_mgr = WaveformManager(self.configs)
        data = (1, 1)
        points = torch.rand(data)
        plateau = waveform_mgr.points_to_plateaus(points)
        point_value = points.tolist()[0]
        plateau_values = plateau.tolist()
        self.assertEqual(
            point_value, plateau_values[random.randint(0, len(plateau_values) - 1)]
        )
        self.assertEqual(point_value, plateau_values[0])

    def runTest(self):
        self.full_check((1, 1))
        self.full_check((10, 1))
        self.full_check((100, 1))
        self.full_check((10, 2))
        self.full_check((100, 7))
        self.test_init()
        self.test_generate_mask_base()
        self.test_expand()
        self.test_points_to_waveform()
        self.test_points_to_plateaus()


if __name__ == "__main__":

    unittest.main()
