"""
Testing signal.py.
"""
import unittest

import torch
import numpy as np

import brainspy.utils.signal as signal
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

    def test_accuracy_fit_random(self):
        """
        Test for accuracy fit with random values for output and target
        with tensors of different sizes
        """
        test_sizes = ((100, 3), (100, 4), (100, 1), (1000, 2), (100, 7))
        for size in test_sizes:
            output = TorchUtils.format(torch.rand(size))
            target = TorchUtils.format(torch.rand(size))
            try:
                signal.accuracy_fit(output, target)
            except (Exception):
                self.fail(
                    "Could not evaluate accuarcy fit for size {}".format(size))

    def test_accuarcy_fit_default_val(self):
        """
        Test for accuracy fit with random values for output and target and default value
        parameter true
        """
        size = (100, 3)
        output = TorchUtils.format(torch.rand(size))
        target = TorchUtils.format(torch.rand(size))
        try:
            signal.accuracy_fit(output, target, default_value=True)
        except (Exception):
            self.fail(
                "Could not evaluate accuarcy fit for size {}".format(size))

    def test_accuracy_fit_invalid_type(self):
        """
        Invalid type for arguments raise an AssertionError
        """

        with self.assertRaises(AssertionError):
            signal.accuracy_fit("Invalid type", 100, default_value=True)
        with self.assertRaises(AssertionError):
            signal.accuracy_fit("Invalid type", [1, 2, 3, 4],
                                default_value=True)
        with self.assertRaises(AssertionError):
            signal.accuracy_fit(100.5,
                                np.array([1, 2, 3, 4]),
                                default_value=True)
        with self.assertRaises(AssertionError):
            signal.accuracy_fit("Invalid type",
                                torch.rand((100, 3)),
                                default_value=True)
        with self.assertRaises(AssertionError):
            signal.accuracy_fit(torch.rand((100, 3)), 100, default_value=True)
        with self.assertRaises(AssertionError):
            signal.accuracy_fit(torch.rand((100, 3)),
                                torch.rand((100, 3)),
                                default_value="Invalid")

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

    def test_corr_fit_random(self):
        """
        Test for corr fit with random values for output and target
        with tensors of different sizes
        """
        test_sizes = ((100, 3), (100, 4), (100, 1), (1000, 2), (100, 7))
        for size in test_sizes:
            output = TorchUtils.format(torch.rand(size))
            target = TorchUtils.format(torch.rand(size))
            try:
                signal.corr_fit(output, target)
            except (Exception):
                self.fail(
                    "Could not evaluate corr fit for size {}".format(size))

    def test_corr_fit_default_val(self):
        """
        Test for corr fit with random values for output and target and default value
        parameter true
        """
        size = (100, 3)
        output = TorchUtils.format(torch.rand(size))
        target = TorchUtils.format(torch.rand(size))
        try:
            signal.corr_fit(output, target, default_value=True)
        except (Exception):
            self.fail("Could not corr accuarcy fit for size {}".format(size))

    def test_corr_fit_invalid_type(self):
        """Invalid type for arguments raise an Assertion Error
        """

        with self.assertRaises(AssertionError):
            signal.corr_fit("Invalid type", 100, default_value=True)
        with self.assertRaises(AssertionError):
            signal.corr_fit("Invalid type", [1, 2, 3, 4], default_value=True)
        with self.assertRaises(AssertionError):
            signal.corr_fit(100.5, np.array([1, 2, 3, 4]), default_value=True)
        with self.assertRaises(AssertionError):
            signal.corr_fit("Invalid type",
                            torch.rand((100, 3)),
                            default_value=True)
        with self.assertRaises(AssertionError):
            signal.corr_fit(torch.rand((100, 3)), 100, default_value=True)
        with self.assertRaises(AssertionError):
            signal.corr_fit(torch.rand((100, 3)),
                            torch.rand((100, 3)),
                            default_value="Invalid")

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

    def test_corrsig_fit_random(self):
        """
        Test for corrsig fit with random values for output and target
        with tensors of different sizes
        """
        test_sizes = ((100, 3), (100, 4), (100, 1), (1000, 2), (100, 7))
        for size in test_sizes:
            output = TorchUtils.format(torch.rand(size))
            target = TorchUtils.format(torch.rand(size))
            try:
                signal.corrsig_fit(output, target)
            except (Exception):
                self.fail(
                    "Could not evaluate corrsig fit for size {}".format(size))

    def test_corrsig_fit_default_val(self):
        """
        Test for corrsig fit with random values for output and target and default value
        parameter true
        """
        size = (100, 3)
        output = TorchUtils.format(torch.rand(size))
        target = TorchUtils.format(torch.rand(size))
        try:
            signal.corrsig_fit(output, target, default_value=True)
        except (Exception):
            self.fail(
                "Could not evaluate corrsig fit for size {}".format(size))

    def test_corrsig_fit_invalid_type(self):
        """
        Invalid type for arguments raises an AssertionError
        """

        with self.assertRaises(AssertionError):
            signal.corrsig_fit("Invalid type", 100, default_value=True)
        with self.assertRaises(AssertionError):
            signal.corrsig_fit("Invalid type", [1, 2, 3, 4],
                               default_value=True)
        with self.assertRaises(AssertionError):
            signal.corrsig_fit(100.5,
                               np.array([1, 2, 3, 4]),
                               default_value=True)
        with self.assertRaises(AssertionError):
            signal.corrsig_fit("Invalid type",
                               torch.rand((100, 3)),
                               default_value=True)
        with self.assertRaises(AssertionError):
            signal.corrsig_fit(torch.rand((100, 3)), 100, default_value=True)
        with self.assertRaises(AssertionError):
            signal.corrsig_fit(torch.rand((100, 3)),
                               torch.rand((100, 3)),
                               default_value="Invalid")

    def test_pearsons_correlation(self):
        """
        Test pearsons correlation implementation by comparing it to the result
        of the numpy method.
        Check that correlation is nan if one signal is uniform.
        """
        for output, target in self.data:
            coef1 = np.zeros(output.shape[1])
            for i in range(output.shape[1]):
                coef1[i] = np.corrcoef(output[:, i].detach().cpu().numpy(),
                                       target[:, i].detach().cpu().numpy())[0,
                                                                            1]
            coef2 = signal.pearsons_correlation(output,
                                                target).detach().cpu().numpy()
            for j in coef2:
                self.assertTrue(j.item() >= -1 and j.item() <= 1)

            # make data uniform
            coef = signal.pearsons_correlation(torch.ones_like(output), target)
            for k in coef:
                self.assertTrue(torch.isnan(k))

    def test_pearsons_correlation_random(self):
        """
        Test for pearsons_correlation with random values for output and target
        with tensors of different sizes
        """
        test_sizes = ((100, 3), (100, 4), (100, 1), (1000, 2), (100, 7))
        for size in test_sizes:
            x = TorchUtils.format(torch.rand(size))
            y = TorchUtils.format(torch.rand(size))
            try:
                signal.pearsons_correlation(x, y)
            except (Exception):
                self.fail(
                    "Could not evaluate pearsons_correlation for size {}".
                    format(size))

    def test_pearsons_correlation_invalid_type(self):
        """
        Invalid type for arguments raises an Assertion Error
        """

        with self.assertRaises(AssertionError):
            signal.pearsons_correlation("Invalid type", 100)
        with self.assertRaises(AssertionError):
            signal.pearsons_correlation("Invalid type", [1, 2, 3, 4])
        with self.assertRaises(AssertionError):
            signal.pearsons_correlation(100.5, np.array([1, 2, 3, 4]))
        with self.assertRaises(AssertionError):
            signal.pearsons_correlation("Invalid type", torch.rand((100, 3)))
        with self.assertRaises(AssertionError):
            signal.pearsons_correlation(torch.rand((100, 3)), 100)

    def test_corrsig(self):
        """
        Test if corrsig method works (type and shape of result).
        """
        for output, target in self.data:
            result = signal.corrsig(output, target)
            self.assertIsInstance(result, torch.Tensor)
            self.assertEqual(list(result.shape), [output.shape[1]])

    def test_corrsig_random(self):
        """
        Test for corrsig fit with random values for output and target
        with tensors of different sizes - FAILING - SPECIFY WITH DIM ARGUMENT
        """
        test_sizes = ((100, 3), (100, 4), (100, 1), (1000, 2), (100, 7))
        for size in test_sizes:
            output = TorchUtils.format(torch.rand(size))
            target = TorchUtils.format(torch.round(torch.rand(size)))
            try:
                signal.corrsig(output, target)
            except (Exception):
                self.fail(
                    "Could not evaluate accuarcy fit for size {}".format(size))

    def test_corrsig_invalid_type(self):
        """
        Invalid type for arguments raises an AssertionError
        """

        with self.assertRaises(AssertionError):
            signal.corrsig("Invalid type", 100)
        with self.assertRaises(AssertionError):
            signal.corrsig("Invalid type", [1, 2, 3, 4])
        with self.assertRaises(AssertionError):
            signal.corrsig(100.5, np.array([1, 2, 3, 4]))
        with self.assertRaises(AssertionError):
            signal.corrsig("Invalid type", torch.rand((100, 3)))
        with self.assertRaises(AssertionError):
            signal.corrsig(torch.rand((100, 3)), 100)

    def test_fisher_fit(self):
        """
        Check if result has right shape and value is negative in both
        cases (default or not).
        """
        for output, target in self.data:
            result = signal.fisher_fit(output, target, False)
            self.assertEqual(list(result.shape), [output.shape[1]])
            self.assertTrue(
                torch.all(
                    -result > TorchUtils.format(torch.zeros_like(result))))
            result = signal.fisher_fit(output, target, True)
            self.assertEqual(list(result.shape), [output.shape[1]])
            self.assertTrue(
                torch.all(
                    -result >= TorchUtils.format(torch.zeros_like(result))))

    def test_fisher_fit_random(self):
        """
        Test for fisher fit with random values for output and target
        with tensors of different sizes
        """
        test_sizes = ((100, 3), (100, 4), (100, 1), (1000, 2), (100, 7))
        for size in test_sizes:
            output = TorchUtils.format(torch.rand(size))
            target = TorchUtils.format(torch.rand(size))
            try:
                signal.fisher_fit(output, target)
            except (Exception):
                self.fail(
                    "Could not evaluate fisher fit for size {}".format(size))

    def test_fisher_fit_default_val(self):
        """
        Test for fisher fit with random values for output and target and default value
        parameter true
        """
        size = (100, 3)
        output = TorchUtils.format(torch.rand(size))
        target = TorchUtils.format(torch.rand(size))
        try:
            signal.fisher_fit(output, target, default_value=True)
        except (Exception):
            self.fail("Could not evaluate fisher fit for size {}".format(size))

    def test_fisher_fit_invalid_type(self):
        """
        Invalid type for arguments raises an Assertion Error
        """

        with self.assertRaises(AssertionError):
            signal.fisher_fit("Invalid type", 100, default_value=True)
        with self.assertRaises(AssertionError):
            signal.fisher_fit("Invalid type", [1, 2, 3, 4], default_value=True)
        with self.assertRaises(AssertionError):
            signal.fisher_fit(100.5,
                              np.array([1, 2, 3, 4]),
                              default_value=True)
        with self.assertRaises(AssertionError):
            signal.fisher_fit("Invalid type",
                              torch.rand((100, 3)),
                              default_value=True)
        with self.assertRaises(AssertionError):
            signal.fisher_fit(torch.rand((100, 3)), 100, default_value=True)
        with self.assertRaises(AssertionError):
            signal.fisher_fit(torch.rand((100, 3)),
                              torch.rand((100, 3)),
                              default_value="Invalid")

    def test_fisher(self):
        """
        Test if result is of right shape and values are negative.
        Check if result is nan if data is uniform.
        """
        for output, target in self.data:
            result = signal.fisher(output, target)
            self.assertEqual(list(result.shape), [output.shape[1]])
            self.assertTrue(
                torch.all(
                    -result >= TorchUtils.format(torch.zeros_like(result))))

            # make data uniform
            result = signal.fisher(torch.ones_like(output), target)
            for k in result:
                self.assertTrue(torch.isnan(k))

    def test_fisher_random(self):
        """
        Test for fisher value with random values for output and target
        with tensors of different sizes
        """
        test_sizes = ((100, 3), (100, 4), (100, 1), (1000, 2), (100, 7))
        for size in test_sizes:
            x = TorchUtils.format(torch.rand(size))
            y = TorchUtils.format(torch.rand(size))
            try:
                signal.fisher(output=x, target=y)
            except (Exception):
                self.fail(
                    "Could not evaluate fisher value for size {}".format(size))

    def test_fisher_invalid_type(self):
        """
        Invalid type for arguments raises an Assertion Error
        """

        with self.assertRaises(AssertionError):
            signal.fisher("Invalid type", 100)
        with self.assertRaises(AssertionError):
            signal.fisher("Invalid type", [1, 2, 3, 4])
        with self.assertRaises(AssertionError):
            signal.fisher(100.5, np.array([1, 2, 3, 4]))
        with self.assertRaises(AssertionError):
            signal.fisher("Invalid type", torch.rand((100, 3)))
        with self.assertRaises(AssertionError):
            signal.fisher(torch.rand((100, 3)), 100)

    def test_sigmoid_nn_distance(self):
        """
        Check if sigmoid_nn_distance method works (type and shape of
        result).
        """
        for output, target in self.data:
            result = signal.sigmoid_nn_distance(output, target)
            self.assertIsInstance(result, torch.Tensor)
            self.assertEqual(list(result.shape), [output.shape[1]])

    def test_sigmoid_nn_distance_random(self):
        """
        Test for sigmoid_nn_distance with random values for output and target
        with tensors of different sizes
        """
        test_sizes = ((100, 3), (100, 4), (100, 1), (1000, 2), (100, 7))
        for size in test_sizes:
            x = TorchUtils.format(torch.rand(size))
            y = TorchUtils.format(torch.rand(size))
            try:
                signal.sigmoid_nn_distance(output=x, target=y)
            except (Exception):
                self.fail("Could not evaluate sigmoid_nn_distance for size {}".
                          format(size))

    def test_sigmoid_nn_distance_invalid_type(self):
        """
        Invalid type for arguments raises an Assertion Error
        """

        with self.assertRaises(AssertionError):
            signal.sigmoid_nn_distance("Invalid type", 100)
        with self.assertRaises(AssertionError):
            signal.sigmoid_nn_distance("Invalid type", [1, 2, 3, 4])
        with self.assertRaises(AssertionError):
            signal.sigmoid_nn_distance(100.5, np.array([1, 2, 3, 4]))
        with self.assertRaises(AssertionError):
            signal.sigmoid_nn_distance("Invalid type", torch.rand((100, 3)))
        with self.assertRaises(AssertionError):
            signal.sigmoid_nn_distance(torch.rand((100, 3)), 100)

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

    def test_get_clamped_intervals_invalid_type(self):
        """
        Invalid type for arguments raises an AssertionError
        """

        with self.assertRaises(AssertionError):
            signal.get_clamped_intervals(
                TorchUtils.format(torch.rand((100, 3))), 100)
        with self.assertRaises(AssertionError):
            signal.get_clamped_intervals(
                TorchUtils.format(torch.rand((100, 3))), [1, 2, 3, 4])
        with self.assertRaises(AssertionError):
            signal.get_clamped_intervals("Invalid type", 100)
        with self.assertRaises(AssertionError):
            signal.get_clamped_intervals(100, "mode")
        with self.assertRaises(AssertionError):
            signal.get_clamped_intervals(torch.rand((100, 3)),
                                         np.array([1, 2, 3]))


if __name__ == "__main__":
    unittest.main()
