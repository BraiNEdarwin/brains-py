
import unittest
import numpy as np
from bspyproc.utils.waveform import WaveformManager


class WaveformTest(unittest.TestCase):

    def __init__(self, test_name):
        super(WaveformTest, self).__init__()
        configs = {}
        configs['amplitude_lengths'] = 80
        configs['slope_lengths'] = 20
        self.configs = configs
        self.waveform_mgr = WaveformManager(configs)

    def test_point_to_waveform(self):
        points = np.random.rand(10)
        waveform = self.waveform_mgr.point_to_waveform(points)
        processed_points = self.waveform_mgr.waveform_to_points(waveform)
        assert (np.float32(processed_points) == np.float32(points)).all(), "It was not possible to transform back the points into waveforms"

    def test_plateau_to_waveform(self):
        points = np.random.rand(3)
        waveform = self.waveform_mgr.point_to_waveform(points)
        mask = self.waveform_mgr.identify_mask(len(waveform))
        waveform_processed = self.waveform_mgr.plateau_to_waveform(waveform[mask])
        assert (waveform == waveform_processed).all(), "It was not possible to transform back the plateaus into waveforms"
        assert (waveform[mask] == self.waveform_mgr.waveform_to_plateaus(waveform)).all(), "It was not possible to transform waveform into plateaus"
        # print(waveform_processed)

    def runTest(self):
        # self.test_point_to_waveform()
        self.test_plateau_to_waveform()
        return True


if __name__ == '__main__':

    from matplotlib import pyplot as plt
    # suite = unittest.TestSuite()
    # suite.addTest(WaveformTest('test1'))
    # unittest.TextTestRunner().run(suite)
    unittest.main()
