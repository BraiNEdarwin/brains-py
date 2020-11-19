#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 11:33:38 2019

@author: hruiz and ualegre
"""

import torch
from torch import nn
import numpy as np

from brainspy.utils.pytorch import TorchUtils
from brainspy.processors.processor import Processor


class DNPU(nn.Module):
    """"""

    def __init__(self, arg, alpha=1):
        # alpha is the regularisation factor
        # arg can either be a configs dictionary or an instance of a processor
        super(DNPU, self).__init__()
        if isinstance(arg, Processor):
            self.processor = arg
        else:
            self.processor = Processor(arg)
        # self._init_regularization_factor(alpha)
        self._init_dnpu(alpha)

    def _init_dnpu(self, alpha):
        self.alpha = torch.tensor(alpha, device=TorchUtils.get_accelerator_type(), dtype=TorchUtils.get_data_type())

        for (
            params
        ) in (
            self.parameters()
        ):  # Freeze parameters of the neural network of the surrogate model
            params.requires_grad = False
        self._init_bias()

    def _init_bias(self):
        self.control_low = self.processor.get_control_ranges()[:, 0]
        self.control_high = self.processor.get_control_ranges()[:, 1]
        assert any(
            self.control_low < 0
        ), "Min. Voltage is assumed to be negative, but value is positive!"
        assert any(
            self.control_high > 0
        ), "Max. Voltage is assumed to be positive, but value is negative!"
        bias = self.control_low + (
            self.control_high - self.control_low
        ) * torch.rand(1, len(self.processor.control_indices), dtype=TorchUtils.get_data_type(), device=TorchUtils.get_accelerator_type())

        self.bias = nn.Parameter(bias)

    def forward(self, x):
        return self.processor(x, self.bias.expand(x.size()[0], -1))

    def regularizer(self):
        return self.alpha * (
            torch.sum(
                torch.relu(self.control_low - self.bias)
                + torch.relu(self.bias - self.control_high)
            )
        )

    def hw_eval(self, arg):
        self.eval()
        if isinstance(arg, Processor):
            self.processor = arg
        else:
            self.processor.load_processor(arg)
        assert torch.equal(self.control_low.cpu().half(), self.processor.get_control_ranges()[:, 0].cpu().half()), 'Low control voltage ranges for the new processor are different than the control voltage ranges for which the DNPU was trained.'
        assert torch.equal(self.control_high.cpu().half(), self.processor.get_control_ranges()[:, 1].cpu().half()), 'High control voltage ranges for the new processor are different than the control voltage ranges for which the DNPU was trained.'
        # self._init_electrode_info(hw_processor_configs)

    def set_control_voltages(self, bias):
        with torch.no_grad():
            bias = bias.unsqueeze(dim=0)
            assert (
                self.bias.shape == bias.shape
            ), "Control voltages could not be set due to a shape missmatch with regard to the ones already in the model."
            self.bias = torch.nn.Parameter(TorchUtils.format_tensor(bias))

    def get_control_voltages(self):
        return next(self.parameters()).detach()

    def get_control_ranges(self):
        return self.processor.get_control_ranges()

    def get_input_ranges(self):
        return self.processor.get_input_ranges()

    def get_clipping_value(self):
        return self.processor.get_clipping_value()

    def reset(self):
        for k in range(len(self.control_low)):
            # print(f'    resetting control {k} between : {self.control_low[k], self.control_high[k]}')
            self.bias.data[:, k].uniform_(self.control_low[k], self.control_high[k])

    def set_regul_factor(self, alpha):
        self.alpha = alpha

    def close(self):
        self.processor.close()

    def is_hardware(self):
        return self.processor.is_hardware
