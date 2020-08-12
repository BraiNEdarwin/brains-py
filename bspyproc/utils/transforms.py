import torch

from bspyproc.utils.pytorch import TorchUtils


class CurrentToVoltage():
    def __init__(self, v_min, v_max, x_min=-1, x_max=1, cut=True):
        self.scale, self.offset = get_map_to_voltage_vars(v_min, v_max, x_min, x_max)
        self.x_min = x_min
        self.x_max = x_max
        self.cut = cut

    def __call__(self, x):
        if self.cut:
            x = torch.clamp(x, min=self.x_min, max=self.x_max)
        x = (x * self.scale) + self.offset
        return x


def get_map_to_voltage_vars(v_min, v_max, x_min, x_max):
    scale = ((v_min - v_max) / (x_min - x_max))
    offset = v_max - scale * x_max
    return scale, offset
