"""
Testing signal.py.
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
        Create two datasets to use for the tests.
        """
        # make two dimensional data tensors
        self.data = []
        for i in range(20, 50, 10):
            for j in range(1, 5, 1):
                output = TorchUtils.format(torch.rand((i, j)))
                target = TorchUtils.format(torch.rand((i, j)))
                target = torch.round(target)  # target vector is binary
                self.data.append((output, target))

    def test_accuracy_fit(self):
        """
        Check if result has right shape and values are between 0 and 100 in
        both cases (default or not).
        """
        for output, target in self.data:
            result = signal.accuracy_fit(output, target, True)
            self.assertEqual(list(result.shape), [output.shape[1]])
            for i in result:
                self.assertTrue(i.item() >= 0 and i.item() <= 100)
            result = signal.accuracy_fit(output, target, False)
            self.assertEqual(list(result.shape), [output.shape[1]])
            for i in result:
                self.assertTrue(i.item() >= 0 and i.item() <= 100)

    def test_corr_fit(self):
        """
        Check if shape of result is correct and values are between -1 and 1
        in both cases (default or not).
        Pearson correlation more extensively tested in own method.
        """
        for output, target in self.data:
            result = signal.corr_fit(output, target, True)
            self.assertEqual(list(result.shape), [output.shape[1]])
            for i in result:
                self.assertTrue(i.item() >= -1 and i.item() <= 1)
            result = signal.corr_fit(output, target, False)
            self.assertEqual(list(result.shape), [output.shape[1]])
            for i in result:
                self.assertTrue(i.item() >= -1 and i.item() <= 1)

    def test_corrsig_fit(self):
        """
        Test if shape of results is correct and values are between -1 and 1
        in both cases (default or not).
        Test that the answer is nan if target is not binary.
        """
        for output, target in self.data:
            result = signal.corrsig_fit(output, target, True)
            self.assertEqual(list(result.shape), [output.shape[1]])
            for i in result:
                self.assertTrue(i.item() >= -1 and i.item() <= 1)
            result = signal.corrsig_fit(output, target, False)
            self.assertEqual(list(result.shape), [output.shape[1]])
            for i in result:
                self.assertTrue(i.item() >= -1 and i.item() <= 1)

            # make target non-binary
            result = signal.corrsig_fit(output, target * 10)
            for i in result:
                self.assertTrue(torch.isnan(i))

    def test_pearsons_correlation(self):
        """
        Test pearsons correlation implementation by comparing it to the result
        of the numpy method.
        Check that correlation is nan if one signal is uniform.
        """
        for output, target in self.data:
            coef1 = TorchUtils.format(torch.zeros(output.shape[1]))
            for i in range(output.shape[1]):
                coef1[i] = np.corrcoef(output[:, i].numpy(),
                                       target[:, i].numpy())[0, 1]
            coef2 = signal.pearsons_correlation(output, target).numpy()
            self.assertTrue(np.allclose(coef1, coef2))
            for j in coef2:
                self.assertTrue(j.item() >= -1 and j.item() <= 1)

            # make data uniform
            coef = signal.pearsons_correlation(torch.ones_like(output), target)
            for k in coef:
                self.assertTrue(torch.isnan(k))

    def test_corrsig(self):
        """
        Test if corrsig method works (type and shape of result).
        """
        for output, target in self.data:
            result = signal.corrsig(output, target)
            self.assertIsInstance(result, torch.Tensor)
            self.assertEqual(list(result.shape), [output.shape[1]])

    def test_fisher_fit(self):
        """
        Check if result has right shape and value is non-negative in both
        cases (default or not).
        """
        for output, target in self.data:
            result = signal.fisher_fit(output, target, False)
            self.assertEqual(list(result.shape), [output.shape[1]])
            self.assertTrue(
                torch.all(
                    result > TorchUtils.format(torch.zeros_like(result))))
            result = signal.fisher_fit(output, target, True)
            self.assertEqual(list(result.shape), [output.shape[1]])
            self.assertTrue(
                torch.all(
                    result >= TorchUtils.format(torch.zeros_like(result))))

    def test_fisher(self):
        """
        Test if result is of right shape and values are nonnegative.
        Check if result is nan if data is uniform.
        """
        for output, target in self.data:
            result = signal.fisher(output, target)
            self.assertEqual(list(result.shape), [output.shape[1]])
            self.assertTrue(
                torch.all(
                    result >= TorchUtils.format(torch.zeros_like(result))))

            # make data uniform
            result = signal.fisher(torch.ones_like(output), target)
            for k in result:
                self.assertTrue(torch.isnan(k))

    def test_sigmoid_nn_distance(self):
        """
        Check if sigmoid_nn_distance method works (type and shape of
        result).
        """
        for output, target in self.data:
            result = signal.sigmoid_nn_distance(output, target)
            self.assertIsInstance(result, torch.Tensor)
            self.assertEqual(list(result.shape), [output.shape[1]])

    def test_get_clamped_intervals(self):
        """
        Test each mode for output data to see if it runs. Test each mode with
        a smaller example and check result.
        """
        for output, target in self.data:
            result = signal.get_clamped_intervals(output, "single_nn")
            result = signal.get_clamped_intervals(output, "double_nn")
            result = signal.get_clamped_intervals(output, "intervals")
            result = signal.get_clamped_intervals(output, "test")

            output = TorchUtils.format(
                torch.tensor([3.0, 1.0, 8.0, 9.0, 5.0]).unsqueeze(dim=1))
            clamp = [1, 9]
            # ordered is 1, 3, 5, 8, 9
            # attach clamped values on both sides: 1, 1, 3, 5, 8, 9, 9
            # distances are 0, 2, 2, 3, 1, 0 (double)
            # smaller distance for each is 0, 2, 2, 1, 0 (single)
            # sum from both sides is 2, 4, 5, 4, 1 (intervals)
            single_result = TorchUtils.format(
                torch.tensor([0.0, 2.0, 2.0, 1.0, 0.0]).unsqueeze(dim=1))
            double_result = TorchUtils.format(
                torch.tensor([0.0, 2.0, 2.0, 3.0, 1.0, 0.0]).unsqueeze(dim=1))
            intervals_result = TorchUtils.format(
                torch.tensor([2.0, 4.0, 5.0, 4.0, 1.0]).unsqueeze(dim=1))
            result = signal.get_clamped_intervals(output, "single_nn", clamp)
            print(result)
            self.assertTrue(torch.equal(result, single_result))
            result = signal.get_clamped_intervals(output, "double_nn", clamp)
            self.assertTrue(torch.equal(result, double_result))
            result = signal.get_clamped_intervals(output, "intervals", clamp)
            self.assertTrue(torch.equal(result, intervals_result))


if __name__ == "__main__":
    unittest.main()
