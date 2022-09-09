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
    def __init__(self, variance: float):
        """
        Initiate object, set standard deviation.

        Parameters
        ----------
        variance : float
            The variance of the noise. It is typically defined by the root
            mean squared deviation error obtained during the training of a
            surrogate model.
        """
        assert (type(variance) == float or type(variance) == int)
        self.std = torch.tensor(
            [variance],
            device=TorchUtils.get_device(),
            dtype=torch.get_default_dtype(),
        )

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
        assert (type(x) == torch.Tensor)
        return x + (self.std * torch.randn(
            x.shape,
            device=TorchUtils.get_device(),
            dtype=torch.get_default_dtype(),
        ))


def get_noise(configs: dict, **kwargs):
    """
    Get given noise type.

    Example
    -------
    >>> get_noise(noise_type="gaussian", variance=1)
    GaussianNoise

    Parameters
    ----------
    configs: dict
        A dictionary containing the configurations to declare different types
        of noise. The dictionary should at least contain the following keys:
            type : str
                Type of noise to be applied. The only currently implemented
                noise type is 'gaussian'.
            variance:
                The variance of the noise. It is typically defined by
                the root mean squared deviation error obtained during the
                training of a surrogate model.

    **kwargs
        Arguments for the noise.

    Returns
    -------
    noise
        A noise generating object. Will be None if input is None or not
        recognized.

    Raises
    ------
    UserWarning
        If the string given does not correspond to an implemented noise type.
    """

    if configs is not None:
        assert (type(configs) == dict)
        if 'type' in configs and configs["type"] == "gaussian":
            return GaussianNoise(configs["variance"])
        else:
            warnings.warn("No noise is being simulated for the model.")
            return None
    return None
