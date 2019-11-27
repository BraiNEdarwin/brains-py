'''Author: HC Ruiz Euler and Unai Alegre-Ibarra; 
DNPU based network of devices to solve complex tasks 25/10/2019
'''

import torch
import numpy as np
import torch.nn as nn
from bspyproc.utils.pytorch import TorchUtils
# from bspyproc.processors.simulation.dopanet import DNPU


class DNPUArchitecture(nn.Module):
    def __init__(self, configs):
        super().__init__()
        # self.in_dict = in_dict
        self.conversion_offset = torch.tensor(-0.6)
        self.offset = self.init_offset(-0.35, 0.7)
        self.scale = self.init_scale(0.1, 1.5)

    def init_offset(self, offset_min, offset_max):
        offset = offset_min + offset_max * np.random.rand(1, 2)
        offset = TorchUtils.get_tensor_from_numpy(offset)
        return nn.Parameter(offset)

    def init_scale(self, scale_min, scale_max):
        scale = scale_min + scale_max * np.random.rand(1)
        scale = TorchUtils.get_tensor_from_numpy(scale)
        return nn.Parameter(scale)

    def offset_penalty(self):
        return torch.sum(torch.relu(self.offset_min - self.offset) + torch.relu(self.offset - self.offset_max))

    def scale_penalty(self):
        return torch.sum(torch.relu(self.scale_min - self.scale) + torch.relu(self.scale - self.scale_m))
