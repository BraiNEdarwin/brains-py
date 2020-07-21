import torch
from bspyproc.utils.pytorch import TorchUtils


class NoNoise():
    def __call__(self, x):
        return x


class GaussianNoise():

    def __init__(self, mse):
        self.error = torch.sqrt(TorchUtils.format_tensor(torch.tensor([mse])))

    def __call__(self, x):
        return x + (self.error * TorchUtils.format_tensor(torch.randn(x.shape)))


def get_noise(configs):
    if configs['noise']['type'] == 'gaussian':
        return GaussianNoise(configs['noise']['mse'])
    else:
        return NoNoise()
