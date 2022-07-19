"""
Testing the noise module of the simulation processor.
"""

import torch
import random
import unittest
import numpy as np
from brainspy.utils.pytorch import TorchUtils
from brainspy.processors.simulation.noise.noise import GaussianNoise, get_noise


class NoiseTest(unittest.TestCase):
    """
    Class for testing 'noise.py'.
    """
    def __init__(self, test_name):
        super(NoiseTest, self).__init__()
        self.threshold = 10000

    def test_gaussian_zero(self):
        """
        Check if gaussian noise with 0 standard deviation always returns the
        input.
        """
        mse = 0
        gaussian = GaussianNoise(mse)
        x_shape = (10, 10)
        for i in range(10):
            x = torch.rand(x_shape,
                           device=TorchUtils.get_device(),
                           dtype=torch.get_default_dtype())
            self.assertTrue(torch.equal(x, gaussian(x)))

    def test_gaussian(self):
        """
        Check if gaussian noise returns the same shape of tensor as the input.
        """
        for i in range(100):
            mse = 10 * torch.rand(1).item()  # random mse between 1 and 10
            gaussian = GaussianNoise(mse)
            x_shape = (round(10 * torch.rand(1).item()),
                       round(10 * torch.rand(1).item()))
            # random shape of x, 2D between 1 and 10
            x = torch.rand(x_shape,
                           device=TorchUtils.get_device(),
                           dtype=torch.get_default_dtype())
            self.assertEqual(x.shape, gaussian(x).shape)

    def test_init(self):
        """
        Initialize the GaussianNoise object with a random value positive or negative
        raises no errors
        """
        try:
            GaussianNoise(random.randint(-self.threshold, self.threshold))
        except (Exception):
            self.fail(
                "Couldn't initialize GaussianNoise class with the values provided"
            )

    def test_init_fail(self):
        """
        Invalid type for variance raises a AssertionError
        """
        with self.assertRaises(AssertionError):
            GaussianNoise(None)
        with self.assertRaises(AssertionError):
            GaussianNoise("String type")
        with self.assertRaises(AssertionError):
            GaussianNoise(np.array([1, 2, 3]))
        with self.assertRaises(AssertionError):
            GaussianNoise([1, 2, 3, 4])

    def test_call(self):
        """
        Using the _call_ function on the gaussian noise with a random torch tensor
        everytime and testing with differnt sizes
        """
        gaussian = GaussianNoise(
            random.randint(-self.threshold, self.threshold))
        test_sizes = ((1, 1), (10, 1), (100, 1), (10, 2), (100, 7))
        for size in test_sizes:
            for i in range(100):
                try:
                    gaussian(
                        torch.rand(size,
                                   device=TorchUtils.get_device(),
                                   dtype=torch.get_default_dtype()))
                except (Exception):
                    self.fail(
                        "Couldn't use the call function with this test size and values"
                    )

    def test_call_fail(self):
        """
        The __call__ function fails and raises an assertion error if
        an invalid type is provided
        """
        gaussian = GaussianNoise(
            random.randint(-self.threshold, self.threshold))

        with self.assertRaises(AssertionError):
            gaussian(None)
        with self.assertRaises(AssertionError):
            gaussian("String type")
        with self.assertRaises(AssertionError):
            gaussian([1, 2, 3, 4, 5])
        with self.assertRaises(AssertionError):
            gaussian(np.array([1, 2, 3]))

    def test_get_noise(self):
        """
        Tesing the get_noise method to return the correct object
        """
        configs = {
            "type": "gaussian",
            "variance": random.randint(-self.threshold, self.threshold)
        }
        noise = get_noise(configs=configs)
        self.assertIsInstance(noise, GaussianNoise)

    def test_get_noise_none(self):
        """
        Invalid configurations for the get_noise method
        returns a none type object
        """
        configs = {"type": "test", "variance": 2}
        noise = get_noise(configs=configs)
        self.assertIsNone(noise)

        configs = {"type": None}
        noise = get_noise(configs=configs)
        self.assertIsNone(noise)

        noise = get_noise(configs=None)
        self.assertIsNone(noise)

    def test_get_noise_fail(self):
        """
        Invalid argument for the method get_noise raises an AssertionError
        """
        with self.assertRaises(AssertionError):
            get_noise("String")
        with self.assertRaises(AssertionError):
            get_noise([1, 2, 3, 4])
        with self.assertRaises(AssertionError):
            get_noise(1000)
        with self.assertRaises(AssertionError):
            get_noise(np.array([1, 2, 3, 4]))

    def runTest(self):
        self.test_gaussian_zero()
        self.test_gaussian()
        self.test_call()
        self.test_call_fail()
        self.test_init()
        self.test_init_fail()
        self.test_get_noise()
        self.test_get_noise_none()
        self.test_get_noise_fail()


if __name__ == "__main__":
    unittest.main()
