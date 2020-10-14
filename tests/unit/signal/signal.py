import torch
import numpy as np

import unittest2 as unittest
from brainspy.algorithms.modules.signal import pearsons_correlation


class SignalTests(unittest.TestCase):
    def __init__(self, test_name):
        super(SignalTests, self).__init__()

    def is_equal_to_numpy(self):
        a = np.random.rand(25)
        b = np.random.rand(25)

        # Same matrix as torch.Tensor:
        at = torch.from_numpy(a)
        bt = torch.from_numpy(b)

        coef1 = np.corrcoef(a, b)
        coef2 = pearsons_correlation(at, bt)
        eq = np.allclose(coef1[0, 1], coef2.cpu().numpy())
        print("Numpy & Torch complex covariance results equal? > {}".format(eq))
        return eq

    def runTest(self):
        return self.is_equal_to_numpy()


if __name__ == "__main__":

    from matplotlib import pyplot as plt

    suite = unittest.TestSuite()
    suite.addTest(SignalTests("test1"))
    unittest.TextTestRunner().run(suite)
