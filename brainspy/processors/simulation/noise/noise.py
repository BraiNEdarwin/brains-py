import torch
from brainspy.utils.pytorch import TorchUtils


class NoNoise:
    def __call__(self, x):
        return x


class GaussianNoise:
    def __init__(self, mse):
        self.std = torch.sqrt(TorchUtils.format_tensor(torch.tensor([mse])))

    def __call__(self, x):
        return x + (self.std * TorchUtils.format_tensor(torch.randn(x.shape)))


def get_noise(configs):
    if "noise" not in configs:
        return NoNoise()
    elif configs["noise"]["type"] == "gaussian":
        return GaussianNoise(configs["noise"]["mse"])
    else:
        print(
            "Warning: Noise configuration not recognised. No noise is being simulated for the model."
        )
        return NoNoise()
