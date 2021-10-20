"""
Module for testing transforms.py.
"""

import unittest
import torch
import random
import numpy as np
import brainspy.utils.transforms as transforms


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

    # def test_line_extreme(self):
    #     """
    #     Test scale and offset, and evaluation at a point, of a line.
    #     Extreme case: x_min is larger than x_max.
    #     """
    #     x_min = 2
    #     y_min = 0
    #     x_max = 1
    #     y_max = 1
    #     # This is the line y = 2 - x.
    #     x_val = 3
    #     offset = transforms.get_offset(y_min=y_min,
    #                                    y_max=y_max,
    #                                    x_min=x_min,
    #                                    x_max=x_max)
    #     scale = transforms.get_scale(y_min=y_min,
    #                                  y_max=y_max,
    #                                  x_min=x_min,
    #                                  x_max=x_max)
    #     self.assertEqual(offset, 2)
    #     self.assertEqual(scale, -1)
    #     value = transforms.linear_transform(x_val=x_val,
    #                                         y_min=y_min,
    #                                         y_max=y_max,
    #                                         x_min=x_min,
    #                                         x_max=x_max)
    #     self.assertEqual(value, -1)

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

    def test_linear_transform(self):
        """
        Testing the function - linear_transform with random values for for all arguments
        """
        x_min = random.randint(-10000, 100000)
        y_min = random.randint(-10000, 100000)
        x_max = random.randint(x_min + 1, 100002)
        y_max = random.randint(y_min + 1, 100002)
        x_val = random.randint(-10000, 100000)
        try:
            transforms.linear_transform(x_val=x_val,
                                        y_min=y_min,
                                        y_max=y_max,
                                        x_min=x_min,
                                        x_max=x_max)
        except (Exception):
            self.fail("Exception was raised")

        x_min = torch.randint(1, 10, (2, 2))
        y_min = torch.randint(1, 10, (2, 2))
        x_max = torch.randint(11, 20, (2, 2))
        y_max = torch.randint(11, 20, (2, 2))
        x_val = torch.rand(2, 2)
        try:
            transforms.linear_transform(x_val=x_val,
                                        y_min=y_min,
                                        y_max=y_max,
                                        x_min=x_min,
                                        x_max=x_max)
        except (Exception):
            self.fail("Exception was raised")

    def test_linear_transform_min_max(self):
        """
        Test for linear transform with min > max
        """
        x_max = random.randint(-10000, 100000)
        y_max = random.randint(-10000, 100000)
        x_min = random.randint(x_max + 1, 100002)
        y_min = random.randint(y_max + 1, 100002)
        x_val = random.randint(-10000, 100000)
        with self.assertRaises(AssertionError):
            transforms.linear_transform(x_val=x_val,
                                        y_min=y_min,
                                        y_max=y_max,
                                        x_min=x_min,
                                        x_max=x_max)

    def test_linear_transform_fail(self):
        """
        Invalid type for arguments raises TypeError
        """
        x_min = random.randint(-10000, 100000)
        y_min = random.randint(-10000, 100000)
        x_max = random.randint(x_min + 1, 100002)
        y_max = random.randint(y_min + 1, 100002)
        x_val = None
        with self.assertRaises(TypeError):
            transforms.linear_transform(x_val=x_val,
                                        y_min=y_min,
                                        y_max=y_max,
                                        x_min=x_min,
                                        x_max=x_max)

        x_min = random.randint(-10000, 100000)
        y_min = random.randint(-10000, 100000)
        x_max = "String value"
        y_max = random.randint(y_min + 1, 100002)
        x_val = random.randint(-10000, 100000)
        with self.assertRaises(TypeError):
            transforms.linear_transform(x_val=x_val,
                                        y_min=y_min,
                                        y_max=y_max,
                                        x_min=x_min,
                                        x_max=x_max)

        x_min = [1, 2, 3, 4, 5]
        y_min = random.randint(-10000, 100000)
        x_max = random.randint(1, 100002)
        y_max = random.randint(y_min + 1, 100002)
        x_val = random.randint(-10000, 100000)
        with self.assertRaises(TypeError):
            transforms.linear_transform(x_val=x_val,
                                        y_min=y_min,
                                        y_max=y_max,
                                        x_min=x_min,
                                        x_max=x_max)

    def test_get_offset(self):
        """
        Testing the function - get_offset with random values for for all arguments
        """
        x_min = random.randint(-10000, 100000)
        y_min = random.randint(-10000, 100000)
        x_max = random.randint(x_min + 1, 100002)
        y_max = random.randint(y_min + 1, 100002)
        try:
            transforms.get_offset(y_min=y_min,
                                  y_max=y_max,
                                  x_min=x_min,
                                  x_max=x_max)
        except ("Exception"):
            self.fail("Exception was raised")

        x_min = torch.rand(2, 2)
        y_min = torch.rand(2, 2)
        x_max = torch.rand(2, 2)
        y_max = torch.rand(2, 2)
        try:
            transforms.get_offset(y_min=y_min,
                                  y_max=y_max,
                                  x_min=x_min,
                                  x_max=x_max)
        except ("Exception"):
            self.fail("Exception was raised")

    def test_get_offset_min_max(self):
        """
        Test for get_offset with min > max
        """
        x_max = random.randint(-10000, 100000)
        y_max = random.randint(-10000, 100000)
        x_min = random.randint(x_max + 1, 100002)
        y_min = random.randint(y_max + 1, 100002)
        with self.assertRaises(AssertionError):
            transforms.get_offset(y_min=y_min,
                                  y_max=y_max,
                                  x_min=x_min,
                                  x_max=x_max)

    def test_get_offset_fail(self):
        """
        Invalid type for arguments raises TypeError
        """
        x_min = random.randint(-10000, 100000)
        y_min = random.randint(-10000, 100000)
        x_max = random.randint(x_min + 1, 100002)
        y_max = [1, 2, 3, 4, 5]
        with self.assertRaises(TypeError):
            transforms.get_offset(y_min=y_min,
                                  y_max=y_max,
                                  x_min=x_min,
                                  x_max=x_max)

        x_min = random.randint(-10000, 100000)
        y_min = random.randint(-10000, 100000)
        x_max = "String val"
        y_max = random.randint(y_min + 1, 100002)
        with self.assertRaises(TypeError):
            transforms.get_offset(y_min=y_min,
                                  y_max=y_max,
                                  x_min=x_min,
                                  x_max=x_max)

        x_min = None
        y_min = random.randint(-10000, 100000)
        x_max = random.randint(1, 100002)
        y_max = random.randint(y_min + 1, 100002)
        with self.assertRaises(TypeError):
            transforms.get_offset(y_min=y_min,
                                  y_max=y_max,
                                  x_min=x_min,
                                  x_max=x_max)

    def test_get_scale(self):
        """
        Testing the function - get_scale with random values for for all arguments
        """
        x_min = random.randint(-10000, 100000)
        y_min = random.randint(-10000, 100000)
        x_max = random.randint(x_min + 1, 100002)
        y_max = random.randint(y_min + 1, 100002)
        try:
            transforms.get_scale(y_min=y_min,
                                 y_max=y_max,
                                 x_min=x_min,
                                 x_max=x_max)
        except ("Exception"):
            self.fail("Exception was raised")

        x_min = torch.rand(2, 2)
        y_min = torch.rand(2, 2)
        x_max = torch.rand(2, 2)
        y_max = torch.rand(2, 2)
        try:
            transforms.get_scale(y_min=y_min,
                                 y_max=y_max,
                                 x_min=x_min,
                                 x_max=x_max)
        except ("Exception"):
            self.fail("Exception was raised")

    def test_get_scale_min_max(self):
        """
        Test for get_scale with min > max
        """
        x_max = random.randint(-10000, 100000)
        y_max = random.randint(-10000, 100000)
        x_min = random.randint(x_max + 1, 100002)
        y_min = random.randint(y_max + 1, 100002)
        with self.assertRaises(AssertionError):
            transforms.get_scale(y_min=y_min,
                                 y_max=y_max,
                                 x_min=x_min,
                                 x_max=x_max)

    def test_get_scale_fail(self):
        """
        Invalid type for arguments raises TypeError
        """
        x_min = random.randint(-10000, 100000)
        y_min = random.randint(-10000, 100000)
        x_max = random.randint(x_min + 1, 100002)
        y_max = [1, 2, 3, 4, 5]
        with self.assertRaises(TypeError):
            transforms.get_scale(y_min=y_min,
                                 y_max=y_max,
                                 x_min=x_min,
                                 x_max=x_max)

        x_min = random.randint(-10000, 100000)
        y_min = random.randint(-10000, 100000)
        x_max = "String val"
        y_max = random.randint(y_min + 1, 100002)
        with self.assertRaises(TypeError):
            transforms.get_scale(y_min=y_min,
                                 y_max=y_max,
                                 x_min=x_min,
                                 x_max=x_max)

        x_min = None
        y_min = random.randint(-10000, 100000)
        x_max = random.randint(1, 100002)
        y_max = random.randint(y_min + 1, 100002)
        with self.assertRaises(TypeError):
            transforms.get_scale(y_min=y_min,
                                 y_max=y_max,
                                 x_min=x_min,
                                 x_max=x_max)

    def test_get_linear_transform_constants(self):
        """
        Testing the function - get_linear_transform_constants with random values for for all
        arguments
        """
        x_min = random.randint(-10000, 100000)
        y_min = random.randint(-10000, 100000)
        x_max = random.randint(x_min + 1, 100002)
        y_max = random.randint(y_min + 1, 100002)
        try:
            transforms.get_linear_transform_constants(y_min=y_min,
                                                      y_max=y_max,
                                                      x_min=x_min,
                                                      x_max=x_max)
        except ("Exception"):
            self.fail("Exception was raised")

        x_min = torch.rand(2, 2)
        y_min = torch.rand(2, 2)
        x_max = torch.rand(2, 2)
        y_max = torch.rand(2, 2)
        try:
            transforms.get_linear_transform_constants(y_min=y_min,
                                                      y_max=y_max,
                                                      x_min=x_min,
                                                      x_max=x_max)
        except ("Exception"):
            self.fail("Exception was raised")

    def test_get_linear_transform_constants_max_min(self):
        """
        Test for get_linear_transform_constants with min > max
        """
        x_max = random.randint(-10000, 100000)
        y_max = random.randint(-10000, 100000)
        x_min = random.randint(x_max + 1, 100002)
        y_min = random.randint(y_max + 1, 100002)
        with self.assertRaises(AssertionError):
            transforms.get_linear_transform_constants(y_min=y_min,
                                                      y_max=y_max,
                                                      x_min=x_min,
                                                      x_max=x_max)

    def test_get_linear_transform_constants_fail(self):
        """
        Invalid type for arguments raises TypeError
        """
        x_min = random.randint(-10000, 100000)
        y_min = random.randint(-10000, 100000)
        x_max = random.randint(x_min + 1, 100002)
        y_max = [1, 2, 3, 4, 5]
        with self.assertRaises(TypeError):
            transforms.get_linear_transform_constants(y_min=y_min,
                                                      y_max=y_max,
                                                      x_min=x_min,
                                                      x_max=x_max)

        x_min = random.randint(-10000, 100000)
        y_min = random.randint(-10000, 100000)
        x_max = "String val"
        y_max = random.randint(y_min + 1, 100002)
        with self.assertRaises(TypeError):
            transforms.get_linear_transform_constants(y_min=y_min,
                                                      y_max=y_max,
                                                      x_min=x_min,
                                                      x_max=x_max)

        x_min = None
        y_min = random.randint(-10000, 100000)
        x_max = random.randint(1, 100002)
        y_max = random.randint(y_min + 1, 100002)
        with self.assertRaises(TypeError):
            transforms.get_linear_transform_constants(y_min=y_min,
                                                      y_max=y_max,
                                                      x_min=x_min,
                                                      x_max=x_max)

    def test_format_input_ranges(self):
        """
        Test the format_input_ranges method.
        """
        t = torch.tensor([[1, 2, 3], [4, 5, 6]])
        result = transforms.format_input_ranges(7, 8, t)
        self.assertTrue(
            torch.equal(result, torch.tensor([[7, 7, 7], [8, 8, 8]])))

    def test_format_input_ranges_random(self):
        """
        Test the format_input_ranges method with random values.
        """
        tensor = torch.rand(2, 2)
        try:
            transforms.format_input_ranges(random.randint(-10000, 100000),
                                           random.randint(-10000, 100000),
                                           tensor)
        except (Exception):
            self.fail("Exception was raised")

    def test_format_input_ranges_fail(self):
        """
        Invalid type raises TypeError
        """
        tensor = torch.rand(2, 2)
        with self.assertRaises(TypeError):
            transforms.format_input_ranges(random.randint(-10000, 100000),
                                           random.randint(-10000, 100000),
                                           [1, 2, 3, 4])
        with self.assertRaises(TypeError):
            transforms.format_input_ranges(None,
                                           random.randint(-10000,
                                                          100000), tensor)
        with self.assertRaises(TypeError):
            transforms.format_input_ranges(random.randint(-10000, 100000),
                                           None, [1, 2, 3, 4])
        with self.assertRaises(TypeError):
            transforms.format_input_ranges("String type",
                                           random.randint(-10000, 100000),
                                           [1, 2, 3, 4])


if __name__ == "__main__":
    unittest.main()
