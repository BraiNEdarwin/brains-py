"""
Testing the noise module of the simulation processor.
"""

import unittest

import torch

from brainspy.processors.simulation.noise.noise import GaussianNoise, get_noise
from brainspy.utils.pytorch import TorchUtils


class TransformsTest(unittest.TestCase):
    """
    Class for testing 'noise.py'.
    """
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

    def test_get_noise(self):
        """
        Check if get_noise method returns a noise object, or None if input is
        None.
        """
        noise = get_noise(noise_type="gaussian", mse=1)
        self.assertIsInstance(noise, GaussianNoise)
        noise = get_noise(noise_type="test", mse=2)
        self.assertIsNone(noise)
        noise = get_noise(noise_type=None)
        self.assertIsNone(noise)


if __name__ == "__main__":
    unittest.main()
