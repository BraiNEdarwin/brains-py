"""
Module for testing transforms.py.
"""

import unittest
import torch
import random
import brainspy.utils.transforms as transforms


class TransformsTest(unittest.TestCase):
    """
    Class for testing 'transforms.py'.
    """
    def __init__(self, test_name):
        super(TransformsTest, self).__init__()
        self.threshold = 10000

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
        x_min = random.randint(-self.threshold, self.threshold)
        y_min = random.randint(-self.threshold, self.threshold)
        x_max = random.randint(x_min + 1, self.threshold + 2)
        y_max = random.randint(y_min + 1, self.threshold + 2)
        x_val = random.randint(-self.threshold, self.threshold)
        try:
            transforms.linear_transform(x_val=x_val,
                                        y_min=y_min,
                                        y_max=y_max,
                                        x_min=x_min,
                                        x_max=x_max)
        except (Exception):
            self.fail(
                "Couldn't perform linear transform with the values provided")

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
            self.fail(
                "Couldn't perform linear transform with the values provided")

    def test_linear_transform_min_max(self):
        """
        Test for linear transform with x_max > x_min and y_min > y_max
        which invokes the get_linear_transform_constants function and raises an Assertion error
        """
        x_max = random.randint(-self.threshold, self.threshold)
        y_max = random.randint(-self.threshold, self.threshold)
        x_min = random.randint(x_max + 1, self.threshold + 2)
        y_min = random.randint(y_max + 1, self.threshold + 2)
        x_val = random.randint(-self.threshold, self.threshold)
        with self.assertRaises(AssertionError):
            transforms.linear_transform(x_val=x_val,
                                        y_min=y_min,
                                        y_max=y_max,
                                        x_min=x_min,
                                        x_max=x_max)

        x_min = torch.randint(11, 20, (2, 2))
        y_min = torch.randint(11, 20, (2, 2))
        x_max = torch.randint(1, 10, (2, 2))
        y_max = torch.randint(1, 10, (2, 2))
        x_val = random.randint(-self.threshold, self.threshold)
        with self.assertRaises(AssertionError):
            transforms.linear_transform(x_val=x_val,
                                        y_min=y_min,
                                        y_max=y_max,
                                        x_min=x_min,
                                        x_max=x_max)

        x_min = torch.randint(11, 20, (2, 2))
        y_min = torch.randint(11, 20, (2, 2))
        x_max = 5
        y_max = 5
        x_val = random.randint(-self.threshold, self.threshold)
        with self.assertRaises(AssertionError):
            transforms.linear_transform(x_val=x_val,
                                        y_min=y_min,
                                        y_max=y_max,
                                        x_min=x_min,
                                        x_max=x_max)

        x_min = 2
        y_min = 0
        x_max = 1
        y_max = 1
        x_val = random.randint(-self.threshold, self.threshold)
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
        x_min = random.randint(-self.threshold, self.threshold)
        y_min = random.randint(-self.threshold, self.threshold)
        x_max = random.randint(x_min + 1, self.threshold + 2)
        y_max = random.randint(y_min + 1, self.threshold + 2)
        x_val = None
        with self.assertRaises(TypeError):
            transforms.linear_transform(x_val=x_val,
                                        y_min=y_min,
                                        y_max=y_max,
                                        x_min=x_min,
                                        x_max=x_max)

        x_min = random.randint(-self.threshold, self.threshold)
        y_min = random.randint(-self.threshold, self.threshold)
        x_max = "String value"
        y_max = random.randint(y_min + 1, self.threshold + 2)
        x_val = random.randint(-self.threshold, self.threshold)
        with self.assertRaises(TypeError):
            transforms.linear_transform(x_val=x_val,
                                        y_min=y_min,
                                        y_max=y_max,
                                        x_min=x_min,
                                        x_max=x_max)

        x_min = [1, 2, 3, 4, 5]
        y_min = random.randint(-self.threshold, self.threshold)
        x_max = random.randint(1, self.threshold + 2)
        y_max = random.randint(y_min + 1, self.threshold + 2)
        x_val = random.randint(-self.threshold, self.threshold)
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
        x_min = random.randint(-self.threshold, self.threshold)
        y_min = random.randint(-self.threshold, self.threshold)
        x_max = random.randint(x_min + 1, self.threshold + 2)
        y_max = random.randint(y_min + 1, self.threshold + 2)
        try:
            transforms.get_offset(y_min=y_min,
                                  y_max=y_max,
                                  x_min=x_min,
                                  x_max=x_max)
        except ("Exception"):
            self.fail("Offset cannot be generated with the values provided")

        x_min = torch.randint(1, 10, (2, 2))
        y_min = torch.randint(1, 10, (2, 2))
        x_max = torch.randint(11, 20, (2, 2))
        y_max = torch.randint(11, 20, (2, 2))
        try:
            transforms.get_offset(y_min=y_min,
                                  y_max=y_max,
                                  x_min=x_min,
                                  x_max=x_max)
        except ("Exception"):
            self.fail("Offset cannot be generated with the values provided")

    def test_get_offset_min_max(self):
        """
        Test for get_offset method with x_min > x_max
        and y_min > y_max raises an assertion error
        """
        x_max = random.randint(-self.threshold, self.threshold)
        y_max = random.randint(-self.threshold, self.threshold)
        x_min = random.randint(x_max + 1, self.threshold + 2)
        y_min = random.randint(y_max + 1, self.threshold + 2)
        with self.assertRaises(AssertionError):
            transforms.get_offset(y_min=y_min,
                                  y_max=y_max,
                                  x_min=x_min,
                                  x_max=x_max)

        x_min = torch.randint(11, 20, (2, 2))
        y_min = torch.randint(11, 20, (2, 2))
        x_max = torch.randint(1, 10, (2, 2))
        y_max = torch.randint(1, 10, (2, 2))
        with self.assertRaises(AssertionError):
            transforms.get_offset(y_min=y_min,
                                  y_max=y_max,
                                  x_min=x_min,
                                  x_max=x_max)

        x_min = torch.randint(11, 20, (2, 2))
        y_min = torch.randint(11, 20, (2, 2))
        x_max = 5
        y_max = 5
        with self.assertRaises(AssertionError):
            transforms.get_offset(y_min=y_min,
                                  y_max=y_max,
                                  x_min=x_min,
                                  x_max=x_max)

        x_min = 2
        y_min = 0
        x_max = 1
        y_max = 1
        with self.assertRaises(AssertionError):
            transforms.get_offset(y_min=y_min,
                                  y_max=y_max,
                                  x_min=x_min,
                                  x_max=x_max)

    def test_get_offset_fail(self):
        """
        Invalid type for arguments raises TypeError
        """
        x_min = random.randint(-self.threshold, self.threshold)
        y_min = random.randint(-self.threshold, self.threshold)
        x_max = random.randint(x_min + 1, self.threshold + 2)
        y_max = [1, 2, 3, 4, 5]
        with self.assertRaises(TypeError):
            transforms.get_offset(y_min=y_min,
                                  y_max=y_max,
                                  x_min=x_min,
                                  x_max=x_max)

        x_min = random.randint(-self.threshold, self.threshold)
        y_min = random.randint(-self.threshold, self.threshold)
        x_max = "String val"
        y_max = random.randint(y_min + 1, self.threshold + 2)
        with self.assertRaises(TypeError):
            transforms.get_offset(y_min=y_min,
                                  y_max=y_max,
                                  x_min=x_min,
                                  x_max=x_max)

        x_min = None
        y_min = random.randint(-self.threshold, self.threshold)
        x_max = random.randint(1, self.threshold + 2)
        y_max = random.randint(y_min + 1, self.threshold + 2)
        with self.assertRaises(TypeError):
            transforms.get_offset(y_min=y_min,
                                  y_max=y_max,
                                  x_min=x_min,
                                  x_max=x_max)

    def test_get_scale(self):
        """
        Testing the function - get_scale with random values for for all arguments
        """
        x_min = random.randint(-self.threshold, self.threshold)
        y_min = random.randint(-self.threshold, self.threshold)
        x_max = random.randint(x_min + 1, self.threshold + 2)
        y_max = random.randint(y_min + 1, self.threshold + 2)
        try:
            transforms.get_scale(y_min=y_min,
                                 y_max=y_max,
                                 x_min=x_min,
                                 x_max=x_max)
        except ("Exception"):
            self.fail("Cannot get scale with the values provided")

        x_min = torch.randint(1, 10, (2, 2))
        y_min = torch.randint(1, 10, (2, 2))
        x_max = torch.randint(11, 20, (2, 2))
        y_max = torch.randint(11, 20, (2, 2))
        try:
            transforms.get_scale(y_min=y_min,
                                 y_max=y_max,
                                 x_min=x_min,
                                 x_max=x_max)
        except ("Exception"):
            self.fail("Cannot get scale with the values provided")

    def test_get_scale_min_max(self):
        """
        Test for get_scale method with x_min > x_max
        and y_min > y_max raises an assertion error
        """
        x_max = random.randint(-self.threshold, self.threshold)
        y_max = random.randint(-self.threshold, self.threshold)
        x_min = random.randint(x_max + 1, self.threshold + 2)
        y_min = random.randint(y_max + 1, self.threshold + 2)
        with self.assertRaises(AssertionError):
            transforms.get_scale(y_min=y_min,
                                 y_max=y_max,
                                 x_min=x_min,
                                 x_max=x_max)

        x_min = torch.randint(11, 20, (2, 2))
        y_min = torch.randint(11, 20, (2, 2))
        x_max = torch.randint(1, 10, (2, 2))
        y_max = torch.randint(1, 10, (2, 2))
        with self.assertRaises(AssertionError):
            transforms.get_scale(y_min=y_min,
                                 y_max=y_max,
                                 x_min=x_min,
                                 x_max=x_max)

        x_min = torch.randint(11, 20, (2, 2))
        y_min = torch.randint(11, 20, (2, 2))
        x_max = 5
        y_max = 5
        with self.assertRaises(AssertionError):
            transforms.get_scale(y_min=y_min,
                                 y_max=y_max,
                                 x_min=x_min,
                                 x_max=x_max)

        x_min = 2
        y_min = 0
        x_max = 1
        y_max = 1
        with self.assertRaises(AssertionError):
            transforms.get_scale(y_min=y_min,
                                 y_max=y_max,
                                 x_min=x_min,
                                 x_max=x_max)

    def test_get_scale_fail(self):
        """
        Invalid type for arguments raises TypeError
        """
        x_min = random.randint(-self.threshold, self.threshold)
        y_min = random.randint(-self.threshold, self.threshold)
        x_max = random.randint(x_min + 1, self.threshold + 2)
        y_max = [1, 2, 3, 4, 5]
        with self.assertRaises(TypeError):
            transforms.get_scale(y_min=y_min,
                                 y_max=y_max,
                                 x_min=x_min,
                                 x_max=x_max)

        x_min = random.randint(-self.threshold, self.threshold)
        y_min = random.randint(-self.threshold, self.threshold)
        x_max = "String val"
        y_max = random.randint(y_min + 1, self.threshold + 2)
        with self.assertRaises(TypeError):
            transforms.get_scale(y_min=y_min,
                                 y_max=y_max,
                                 x_min=x_min,
                                 x_max=x_max)

        x_min = None
        y_min = random.randint(-self.threshold, self.threshold)
        x_max = random.randint(1, self.threshold + 2)
        y_max = random.randint(y_min + 1, self.threshold + 2)
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
        x_min = random.randint(-self.threshold, self.threshold)
        y_min = random.randint(-self.threshold, self.threshold)
        x_max = random.randint(x_min + 1, self.threshold + 2)
        y_max = random.randint(y_min + 1, self.threshold + 2)
        try:
            transforms.get_linear_transform_constants(y_min=y_min,
                                                      y_max=y_max,
                                                      x_min=x_min,
                                                      x_max=x_max)
        except ("Exception"):
            self.fail(
                "Cannot generate scale and offset with the values provided")

        x_min = torch.randint(1, 10, (2, 2))
        y_min = torch.randint(1, 10, (2, 2))
        x_max = torch.randint(11, 20, (2, 2))
        y_max = torch.randint(11, 20, (2, 2))
        try:
            transforms.get_linear_transform_constants(y_min=y_min,
                                                      y_max=y_max,
                                                      x_min=x_min,
                                                      x_max=x_max)
        except ("Exception"):
            self.fail(
                "Cannot generate scale and offset with the values provided")

    def test_get_linear_transform_constants_max_min(self):
        """
        Test for get_linear_transform_constants method with x_min > x_max
        and y_min > y_max raises an assertion error
        """
        x_max = random.randint(-self.threshold, self.threshold)
        y_max = random.randint(-self.threshold, self.threshold)
        x_min = random.randint(x_max + 1, self.threshold + 2)
        y_min = random.randint(y_max + 1, self.threshold + 2)
        with self.assertRaises(AssertionError):
            transforms.get_linear_transform_constants(y_min=y_min,
                                                      y_max=y_max,
                                                      x_min=x_min,
                                                      x_max=x_max)

        x_min = torch.randint(11, 20, (2, 2))
        y_min = torch.randint(11, 20, (2, 2))
        x_max = torch.randint(1, 10, (2, 2))
        y_max = torch.randint(1, 10, (2, 2))
        with self.assertRaises(AssertionError):
            transforms.get_linear_transform_constants(y_min=y_min,
                                                      y_max=y_max,
                                                      x_min=x_min,
                                                      x_max=x_max)

        x_min = torch.randint(11, 20, (2, 2))
        y_min = torch.randint(11, 20, (2, 2))
        x_max = 5
        y_max = 5
        with self.assertRaises(AssertionError):
            transforms.get_linear_transform_constants(y_min=y_min,
                                                      y_max=y_max,
                                                      x_min=x_min,
                                                      x_max=x_max)

        x_min = 2
        y_min = 0
        x_max = 1
        y_max = 1
        with self.assertRaises(AssertionError):
            transforms.get_linear_transform_constants(y_min=y_min,
                                                      y_max=y_max,
                                                      x_min=x_min,
                                                      x_max=x_max)

    def test_get_linear_transform_constants_fail(self):
        """
        Invalid type for arguments raises TypeError
        """
        x_min = random.randint(-self.threshold, self.threshold)
        y_min = random.randint(-self.threshold, self.threshold)
        x_max = random.randint(x_min + 1, self.threshold + 2)
        y_max = [1, 2, 3, 4, 5]
        with self.assertRaises(TypeError):
            transforms.get_linear_transform_constants(y_min=y_min,
                                                      y_max=y_max,
                                                      x_min=x_min,
                                                      x_max=x_max)

        x_min = random.randint(-self.threshold, self.threshold)
        y_min = random.randint(-self.threshold, self.threshold)
        x_max = "String val"
        y_max = random.randint(y_min + 1, self.threshold + 2)
        with self.assertRaises(TypeError):
            transforms.get_linear_transform_constants(y_min=y_min,
                                                      y_max=y_max,
                                                      x_min=x_min,
                                                      x_max=x_max)

        x_min = None
        y_min = random.randint(-self.threshold, self.threshold)
        x_max = random.randint(1, self.threshold + 2)
        y_max = random.randint(y_min + 1, self.threshold + 2)
        with self.assertRaises(TypeError):
            transforms.get_linear_transform_constants(y_min=y_min,
                                                      y_max=y_max,
                                                      x_min=x_min,
                                                      x_max=x_max)

    def runTest(self):
        self.test_line()
        self.test_line_nan()
        self.test_get_linear_transform_constants()
        self.test_get_linear_transform_constants_fail()
        self.test_get_linear_transform_constants_max_min()
        self.test_get_offset()
        self.test_get_offset_fail()
        self.test_get_offset_min_max()
        self.test_get_scale()
        self.test_get_scale_fail()
        self.test_get_scale_min_max()
        self.test_linear_transform()
        self.test_linear_transform_fail()
        self.test_linear_transform_min_max()


if __name__ == "__main__":
    unittest.main()
