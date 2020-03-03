#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 11:33:38 2019

@author: hruiz
"""


import torch
import numpy as np
import torch.nn as nn
from bspyproc.processors.simulation.surrogate import SurrogateModel
from bspyproc.utils.pytorch import TorchUtils
from bspyproc.utils.control import merge_inputs_and_control_voltages_in_torch


class DNPU(SurrogateModel):
    '''
    '''

    def __init__(self, configs):
        super().__init__(configs)
        self.init_electrode_info(configs)
        # Freeze parameters
        for params in self.parameters():
            params.requires_grad = False
        self.init_bias()

    def init_electrode_info(self, configs):
        self.input_no = len(configs['input_indices'])
        self.input_indices = TorchUtils.get_tensor_from_list(configs['input_indices'], torch.int64)
        self.electrode_no = len(self.info['data_info']['input_data']['offset'])
        self.control_voltage_indices = np.delete(np.arange(self.electrode_no), configs['input_indices'])
        self.control_voltage_indices = TorchUtils.get_tensor_from_list(self.control_voltage_indices, torch.int64)  # IndexError: tensors used as indices must be long, byte or bool tensors
        self.control_voltage_no = len(self.control_voltage_indices)

    def init_bias(self):
        self.control_low = self.min_voltage[self.control_voltage_indices]
        self.control_high = self.max_voltage[self.control_voltage_indices]
        assert any(self.control_low < 0), "Min. Voltage is assumed to be negative, but value is positive!"
        assert any(self.control_high > 0), "Max. Voltage is assumed to be positive, but value is negative!"
        bias = self.min_voltage[self.control_voltage_indices] + \
            (self.max_voltage[self.control_voltage_indices] - self.min_voltage[self.control_voltage_indices]) * \
            TorchUtils.get_tensor_from_numpy(np.random.rand(1, self.control_voltage_no))

        self.bias = nn.Parameter(TorchUtils.get_tensor_from_numpy(bias))

    def forward(self, x):
        inp = merge_inputs_and_control_voltages_in_torch(x, self.bias.expand(x.size()[0], -1), self.input_indices, self.control_voltage_indices)
        return self.forward_processed(inp)

    def regularizer(self):
        return torch.sum(torch.relu(self.control_low - self.bias) + torch.relu(self.bias - self.control_high))

    def reset(self):
        for k in range(len(self.control_low)):
            # print(f'    resetting control {k} between : {self.control_low[k], self.control_high[k]}')
            self.bias.data[:, k].uniform_(self.control_low[k], self.control_high[k])

    def get_control_voltages(self):
        return next(self.parameters()).detach()


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    x = 0.5 * np.random.randn(10, 2)
    x = TorchUtils.get_tensor_from_numpy(x)
    target = TorchUtils.get_tensor_from_list([[5]] * 10)
    node = DNPU([0, 4])
    loss = nn.MSELoss()
    optimizer = torch.optim.Adam([{'params': node.parameters()}], lr=0.01)

    LOSS_LIST = []
    CHANGE_PARAMS_NET = []
    CHANGE_PARAMS0 = []

    START_PARAMS = [p.clone().detach() for p in node.parameters()]

    for eps in range(10000):

        optimizer.zero_grad()
        out = node(x)
        if np.isnan(out.data.cpu().numpy()[0]):
            break
        LOSS = loss(out, target) + node.regularizer()
        LOSS.backward()
        optimizer.step()
        LOSS_LIST.append(LOSS.data.cpu().numpy())
        CURRENT_PARAMS = [p.clone().detach() for p in node.parameters()]
        DELTA_PARAMS = [(current - start).sum() for current, start in zip(CURRENT_PARAMS, START_PARAMS)]
        CHANGE_PARAMS0.append(DELTA_PARAMS[0])
        CHANGE_PARAMS_NET.append(sum(DELTA_PARAMS[1:]))

    END_PARAMS = [p.clone().detach() for p in node.parameters()]
    print("CV params at the beginning: \n ", START_PARAMS[0])
    print("CV params at the end: \n", END_PARAMS[0])
    print("Example params at the beginning: \n", START_PARAMS[-1][:8])
    print("Example params at the end: \n", END_PARAMS[-1][:8])
    print("Length of elements in node.parameters(): \n", [len(p) for p in END_PARAMS])
    print("and their shape: \n", [p.shape for p in END_PARAMS])
    print(f'OUTPUT: \n {out.data.cpu()}')

    plt.figure()
    plt.plot(LOSS_LIST)
    plt.title("Loss per epoch")
    plt.show()
    plt.figure()
    plt.plot(CHANGE_PARAMS0)
    plt.plot(CHANGE_PARAMS_NET)
    plt.title("Difference of parameter with initial params")
    plt.legend(["CV params", "Net params"])
    plt.show()
