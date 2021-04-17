"""
Module for generating noise.
"""
import warnings

import torch

from brainspy.utils.pytorch import TorchUtils


class GaussianNoise:
    """
    Class for generating and applying Gaussian noise.
    """
    def __init__(self, mse: float):
        """
        Initiate object, set standard deviation.

        Parameters
        ----------
        mse : float
            The variance, usually obtained by mean squared error.
        """
        self.std = torch.sqrt(
            torch.tensor(
                [mse],
                device=TorchUtils.get_device(),
                dtype=torch.get_default_dtype(),
            ))

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Gaussian noise to a tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input data.

        Returns
        -------
        torch.Tensor
            Output data.
        """
        return x + (self.std * torch.randn(
            x.shape,
            device=TorchUtils.get_device(),
            dtype=torch.get_default_dtype(),
        ))


def get_noise(noise: str, **kwargs):
    """
    Get given noise type.

    Parameters
    ----------
    noise : str
        Type of noise to be applied.
    **kwargs
        Arguments for the noise.

    Returns
    -------
    noise
        A noise generating object. Will be none if input is none or not
        recognized.

    Raises
    ------
    UserWarning
        If the string given does not correspond to an implemented noise type.
    """
    if noise is not None:
        if noise == "gaussian":
            return GaussianNoise(kwargs["mse"])
        else:
            warnings.warn(
                "Noise configuration not recognised. No noise is being "
                "simulated for the model.")
            return None
    return noise
