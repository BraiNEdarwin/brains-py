import torch
from brainspy.utils.pytorch import TorchUtils


class NoNoise:
    def __call__(self, x):
        return x


class GaussianNoise:
    def __init__(self, mse):
        self.std = torch.sqrt(
            torch.tensor(
                [mse],
                device=TorchUtils.get_accelerator_type(),
                dtype=TorchUtils.get_data_type(),
            )
        )

    def __call__(self, x):
        return x + (
            self.std
            * torch.randn(
                x.shape,
                device=TorchUtils.get_accelerator_type(),
                dtype=TorchUtils.get_data_type(),
            )
        )


def get_noise(configs):
    if "noise" not in configs:
        return NoNoise()
    elif configs["noise"]["type"] == "gaussian":
        return GaussianNoise(configs["driver"]["noise"]["mse"])
    else:
        print(
            "Warning: Noise configuration not recognised. No noise is being simulated for the model."
        )
        return NoNoise()
