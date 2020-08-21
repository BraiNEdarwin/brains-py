"""
Created on Fri Aug  7 10:30:02 2020

@author: ualegre
"""

import torch
from torch import nn
import numpy as np

from brainspy.processors.dnpu import DNPU
from brainspy.utils.pytorch import TorchUtils
from brainspy.utils.transforms import CurrentToVoltage


class DNPUBN(nn.Module):
    '''
        v_min value is the minimum voltage value of the electrode of the next dnpu to which the output is going to be connected
        v_max value is the maximum voltage value of the electrode of the next dnpu to which the output is going to be connected 
    '''

    def __init__(self, configs, v_min, v_max, std=1):
        super().__init__()
        self.dnpu = DNPU(configs)
        self.bn = TorchUtils.format_tensor(nn.BatchNorm1d(1, affine=False))
        cut = 2 * std
        self.current_to_voltage = CurrentToVoltage(v_min=v_min, v_max=v_max, x_min=-cut, x_max=cut)

    def forward(self, x):
        x = self.dnpu(x)
        # Cut off values out of the clipping value
        x = torch.clamp(x, min=self.dnpu.processor.clipping_value[0], max=self.dnpu.processor.clipping_value[1])
        # Apply batch normalisation
        x = self.bn(x)
        # Apply current to voltage transformation
        x = self.current_to_voltage(x)
        return x

    def hw_eval(self, hw_processor_configs):
        self.dnpu.hw_eval(hw_processor_configs)
