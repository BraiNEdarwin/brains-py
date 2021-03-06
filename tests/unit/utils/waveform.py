"""
Unit tests for the waveform manager
@author: ualegre
"""
import unittest2 as unittest
import torch
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
            point_no,
            device=TorchUtils.get_accelerator_type(),
            dtype=TorchUtils.get_data_type(),
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

    def runTest(self):
        self.full_check((1, 1))
        self.full_check((10, 1))
        self.full_check((100, 1))
        self.full_check((10, 2))
        self.full_check((100, 7))


if __name__ == "__main__":
    import HtmlTestRunner

    # unittest.main(
    #     testRunner=xmlrunner.XMLTestRunner(output='/home/unai/Documents/3-programming/brains-py/brains-py/test-reports'),
    #     # these make sure that some options that are not applicable
    #     # remain hidden from the help menu.
    #     failfast=False, buffer=False, catchbreak=False)

    unittest.main(
        testRunner=HtmlTestRunner.HTMLTestRunner(
            output="/home/unai/Documents/3-programming/brains-py/brains-py/test-reports"
        )
    )

    # suite = unittest.TestSuite()
    # suite.addTest(WaveformTest("test1"))
    # unittest.TextTestRunner(verbosity=2).run(suite)

    # outfile = open("report.html", "w")
    # runner = HTMLTestRunner.HTMLTestRunner(
    #     stream=outfile,
    #     title='Test Report',
    #     description='This demonstrates the report output by Prasanna.Yelsangikar.'
    # )
