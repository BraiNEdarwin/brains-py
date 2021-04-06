import torch
from brainspy.utils.pytorch import TorchUtils


class GaussianNoise:
    def __init__(self, mse):
        self.std = torch.sqrt(
            torch.tensor(
                [mse],
                device=TorchUtils.get_device(),
                dtype=torch.get_default_dtype(),
            )
        )

    def __call__(self, x):
        return x + (
            self.std
            * torch.randn(
                x.shape,
                device=TorchUtils.get_device(),
                dtype=torch.get_default_dtype(),
            )
        )


def get_noise(noise, **kwargs):
    if noise is not None:
        if noise == "gaussian":
            return GaussianNoise(kwargs["mse"])
        else:
            print(
                "Warning: Noise configuration not recognised. No noise is being simulated for the model."
            )
            return None
    return noise
