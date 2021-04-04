"""
Module for generating noise.
"""

import torch
from brainspy.utils.pytorch import TorchUtils
import warnings


class GaussianNoise:
    """
    Class for generating and applying Gaussian noise.
    """
    def __init__(self, mse):
        """
        Initiate object, set standard deviation.

        Parameters
        ----------
        mse : [type]
            [description]
        """
        self.std = torch.sqrt(
            torch.tensor(
                [mse],
                device=TorchUtils.get_accelerator_type(),
                dtype=TorchUtils.get_data_type(),
            ))

    def __call__(self, x):
        return x + (self.std * torch.randn(
            x.shape,
            device=TorchUtils.get_accelerator_type(),
            dtype=TorchUtils.get_data_type(),
        ))


def get_noise(noise, **kwargs):
    """
    Get given noise type.

    Parameters
    ----------
    noise : [type]
        [description]

    Returns
    -------
    [type]
        [description]
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
