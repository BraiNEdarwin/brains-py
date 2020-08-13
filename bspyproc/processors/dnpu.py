#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 11:33:38 2019

@author: hruiz and ualegre
"""

import torch
from torch import nn
import numpy as np

from bspyproc.utils.pytorch import TorchUtils
from bspyproc.utils.electrodes import merge_electrode_data

from bspyproc.processors.simulation.surrogate import SurrogateModel
from bspyproc.processors.hardware.processor import HardwareProcessor


class DNPU(nn.Module):
    '''
    '''

    def __init__(self, configs):
        super().__init__()
        self._load_processor(configs)
        self._init_electrode_info(configs)
        self._init_regularization_factor(configs)

        for params in self.parameters():  # Freeze parameters of the neural network of the surrogate model
            params.requires_grad = False
        self._init_bias()

    def _load_processor(self, configs):
        if configs['platform'] == 'hardware':
            self.processor = HardwareProcessor(configs)
            self.electrode_no = len(configs["activation_channels"])
        elif configs['platform'] == 'simulation':
            self.processor = SurrogateModel(configs)
            self.electrode_no = len(self.processor.info['data_info']['input_data']['offset'])
        else:
            raise NotImplementedError(f"Platform {configs['platform']} is not recognised. The platform has to be either 'hardware' or 'simulation'")

    def _init_electrode_info(self, configs):
        # self.input_no = len(configs['input_indices'])
        self.input_indices = TorchUtils.get_tensor_from_list(configs['input_indices'], torch.int64)
        self.control_indices = np.delete(np.arange(self.electrode_no), configs['input_indices'])
        self.control_indices = TorchUtils.get_tensor_from_list(self.control_indices, torch.int64)  # IndexError: tensors used as indices must be long, byte or bool tensors

    def _init_regularization_factor(self, configs):
        if 'regularisation_factor' in configs:
            self.alpha = TorchUtils.format_tensor(torch.tensor(configs['regularisation_factor']))
        else:
            print('No regularisation factor set.')
            self.alpha = TorchUtils.format_tensor(torch.tensor([1]))

    def _init_bias(self):
        self.control_low = self.processor.min_voltage[self.control_indices]
        self.control_high = self.processor.max_voltage[self.control_indices]
        assert any(self.control_low < 0), "Min. Voltage is assumed to be negative, but value is positive!"
        assert any(self.control_high > 0), "Max. Voltage is assumed to be positive, but value is negative!"
        bias = self.control_low + \
            (self.control_high - self.control_low) * \
            TorchUtils.get_tensor_from_numpy(np.random.rand(1, len(self.control_indices)))

        self.bias = nn.Parameter(bias)

    def forward(self, x):
        inp = merge_electrode_data(x, self.bias.expand(x.size()[0], -1), self.input_indices, self.control_indices)
        return self.processor(inp)

    def regularizer(self):
        return self.alpha * (torch.sum(torch.relu(self.control_low - self.bias) + torch.relu(self.bias - self.control_high)))

    def reset(self):
        for k in range(len(self.control_low)):
            # print(f'    resetting control {k} between : {self.control_low[k], self.control_high[k]}')
            self.bias.data[:, k].uniform_(self.control_low[k], self.control_high[k])

    def get_input_range(self):
        return self.processor.min_voltage[self.input_indices], self.processor.max_voltage[self.input_indices]

    def get_control_voltages(self):
        return next(self.parameters()).detach()

    def hw_eval(self, hw_processor_configs):
        self.eval()
        self._load_processor(hw_processor_configs)
        self._init_electrode_info(hw_processor_configs)
