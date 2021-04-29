"""
Module for testing transforms.py.
"""

import unittest

import torch
import numpy as np

import brainspy.utils.transforms as transforms
from brainspy.utils.pytorch import TorchUtils


class TransformsTest(unittest.TestCase):
    """
    Class for testing 'transforms.py'.
    """
    def test_line(self):
        """
        Test scale and offset, and evaluation at a point, of a line.
        """
        x_min = 1
        y_min = 1
        x_max = 2
        y_max = 0
        # This is the line y = 2 - x.
        x_val = 3
        offset = transforms.get_offset(y_min=y_min,
                                       y_max=y_max,
                                       x_min=x_min,
                                       x_max=x_max)
        scale = transforms.get_scale(y_min=y_min,
                                     y_max=y_max,
                                     x_min=x_min,
                                     x_max=x_max)
        both = transforms.get_linear_transform_constants(y_min=y_min,
                                                         y_max=y_max,
                                                         x_min=x_min,
                                                         x_max=x_max)
        self.assertEqual(both, (scale, offset))
        self.assertEqual(offset, 2)
        self.assertEqual(scale, -1)
        value = transforms.linear_transform(x_val=x_val,
                                            y_min=y_min,
                                            y_max=y_max,
                                            x_min=x_min,
                                            x_max=x_max)
        self.assertEqual(value, -1)

    def test_line_extreme(self):
        """
        Test scale and offset, and evaluation at a point, of a line.
        Extreme case: x_min is larger than x_max.
        """
        x_min = 2
        y_min = 0
        x_max = 1
        y_max = 1
        # This is the line y = 2 - x.
        x_val = 3
        offset = transforms.get_offset(y_min=y_min,
                                       y_max=y_max,
                                       x_min=x_min,
                                       x_max=x_max)
        scale = transforms.get_scale(y_min=y_min,
                                     y_max=y_max,
                                     x_min=x_min,
                                     x_max=x_max)
        self.assertEqual(offset, 2)
        self.assertEqual(scale, -1)
        value = transforms.linear_transform(x_val=x_val,
                                            y_min=y_min,
                                            y_max=y_max,
                                            x_min=x_min,
                                            x_max=x_max)
        self.assertEqual(value, -1)

    def test_line_multi_dim(self):
        """
        Test scale and offset, and evaluation at a point, of a line.
        Using multi-dimensional data.
        """
        x_min = np.array([1, 0, 0])
        y_min = np.array([1, 0, 1])
        x_max = np.array([2, 1, 1])
        y_max = np.array([0, 1, 1])
        # Lines are y = 2 - x; y = x; y = 1.
        x_val = np.array([3, 3, 3])
        offset = transforms.get_offset(y_min=y_min,
                                       y_max=y_max,
                                       x_min=x_min,
                                       x_max=x_max)
        scale = transforms.get_scale(y_min=y_min,
                                     y_max=y_max,
                                     x_min=x_min,
                                     x_max=x_max)
        value = transforms.linear_transform(x_val=x_val,
                                            y_min=y_min,
                                            y_max=y_max,
                                            x_min=x_min,
                                            x_max=x_max)
        self.assertTrue(np.array_equal(offset, np.array([2, 0, 1])))
        self.assertTrue(np.array_equal(scale, np.array([-1, 1, 0])))
        self.assertTrue(np.array_equal(value, np.array([-1, 3, 1])))

    def test_line_nan(self):
        """
        Test the line transform for x_min = x_max; should raise
        ZeroDivisionError.
        """
        x_min = 1
        y_min = 1
        x_max = 1
        y_max = 2
        self.assertRaises(
            ZeroDivisionError,
            transforms.get_scale,
            y_min=y_min,
            y_max=y_max,
            x_min=x_min,
            x_max=x_max,
        )

    def test_ctv(self):
        """
        Test the CurrentToVoltage class.
        Largely uses the other methods so it does not need to be tested
        extensively (extreme cases etc).
        """
        ctv = transforms.CurrentToVoltage([[0, 1], [1, 2]], [[1, 2], [1, 0]])

        # First transform is line y = x + 1, second is y = x - 2.
        # For the second one input is out of range, so cut needs to work.
        x_value = torch.tensor([[1, 3]], device=TorchUtils.get_device(), dtype=torch.get_default_dtype())

        result = ctv(x_value)
        target = torch.tensor([[2, 0]],
                              device=TorchUtils.get_device(),
                              dtype=torch.get_default_dtype())
        for i in range(target.shape[0]):
            for j in range(target.shape[1]):
                self.assertEqual(result[i][j], target[i][j])

    def test_format_input_ranges(self):
        """
        Test the format_input_ranges method.
        """
        t = torch.tensor([[1, 2, 3], [4, 5, 6]])
        result = transforms.format_input_ranges(7, 8, t)
        self.assertTrue(
            torch.equal(result, torch.tensor([[7, 7, 7], [8, 8, 8]])))


if __name__ == "__main__":
    unittest.main()
