"""
Testing signal.py
"""
import unittest

import torch
import numpy as np

import brainspy.algorithms.modules.signal as signal
from brainspy.utils.pytorch import TorchUtils


class SignalTest(unittest.TestCase):
    """
    Testing each method in signal.py.
    """
    def setUp(self):
        """
        Create two signals to be compared.
        """
        self.output = TorchUtils.format(torch.rand((100, 1)))
        self.target = TorchUtils.format(torch.rand((100, 1)))
        self.target = torch.round(self.target)  # target vector is binary

    def test_accuracy_fit(self):
        """
        Check if separability is between 0 and 100 in both cases.
        """
        acc = signal.accuracy_fit(self.output, self.target, True)
        self.assertTrue(acc >= 0 and acc <= 100)
        acc = signal.accuracy_fit(self.output, self.target, False).item()
        self.assertTrue(acc >= 0 and acc <= 100)

    def test_corr_fit(self):
        """
        Check if similarity is between -1 and 1 in both cases.
        Pearson correlation more extensively tested in own method.
        """
        corr = signal.corr_fit(self.output, self.target, True)
        self.assertTrue(corr >= -1 and corr <= 1)
        corr = signal.corr_fit(self.output, self.target, False).item()
        self.assertTrue(corr >= -1 and corr <= 1)

    def test_corrsig_fit(self):
        """
        Test if similarity is between -1 and 1.
        Test that the answer is nan if target is not binary.
        """
        corr = signal.corrsig_fit(self.output, self.target, True)
        self.assertTrue(corr >= -1 and corr <= 1)
        corr = signal.corrsig_fit(self.output, self.target, False).item()
        self.assertTrue(corr >= -1 and corr <= 1)

        # make target non-binary
        corr = signal.corrsig_fit(self.output, self.target * 10)
        self.assertTrue(torch.isnan(corr))

    def test_pearsons_correlation(self):
        """
        Test pearsons correlation implementation by comparing it to the result
        of the numpy method.
        Check that correlation is nan if one signal is uniform.
        """
        # squeeze first
        a = self.output[:, 0]
        b = self.target[:, 0]
        coef1 = np.corrcoef(a.numpy(), b.numpy())
        coef2 = signal.pearsons_correlation(a, b)
        self.assertTrue(np.allclose(coef1[0, 1], coef2.numpy()))
        self.assertTrue(coef2.item() >= -1 and coef2.item() <= 1)

        a = torch.ones_like(a)
        coef = signal.pearsons_correlation(a, b)
        self.assertTrue(torch.isnan(coef))

    def test_corrsig(self):
        """
        Test if corrsig method works (type and shape of result).
        """
        # squeeze data
        a = self.output[:, 0]
        b = self.target[:, 0]
        result = signal.corrsig(a, b)
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(list(result.shape), [])

    def test_sqrt_corrsig(self):
        """
        Test if sqrt_corrsig method works (type and shape of result).
        """
        # squeeze data
        a = self.output[:, 0]
        b = self.target[:, 0]
        result = signal.sqrt_corrsig(a, b)
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(list(result.shape), [])

    def test_fisher_fit(self):
        """
        TODO test fisher fit
        """
        signal.fisher_fit(self.output, self.target)

    def test_fisher(self):
        """
        TODO test fisher
        """
        signal.fisher(self.output, self.target)

    def test_fisher_added_corr(self):
        """
        TODO test fisher added corr
        """
        signal.fisher_added_corr(self.output, self.target)

    def test_fisher_multipled_corr(self):
        """
        TODO test fisher multiplied corr
        """
        signal.fisher_multipled_corr(self.output, self.target)

    def test_sigmoid_nn_distance(self):
        """
        TODO test sigmoid nn distance
        """
        signal.sigmoid_nn_distance(self.output, self.target)

    def test_get_clamped_intervals(self):
        """
        TODO test get clamped intervals
        """
        pass


if __name__ == "__main__":
    unittest.main()
