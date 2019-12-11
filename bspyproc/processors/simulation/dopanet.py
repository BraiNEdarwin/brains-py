#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 11:33:38 2019

@author: hruiz
"""


import torch
import numpy as np
import torch.nn as nn
from bspyproc.processors.simulation.network import TorchModel
from bspyproc.utils.pytorch import TorchUtils


class DNPU(TorchModel):
    '''
    '''

    def __init__(self, configs):
        super().__init__(configs)
        self.nr_inputs = len(configs['input_indices'])
        self.in_list = TorchUtils.get_tensor_from_list(configs['input_indices'], torch.int64)
        # Freeze parameters
        for params in self.parameters():
            params.requires_grad = False
        # Define learning parameters
        self.nr_electodes = len(self.info['offset'])
        self.indx_cv = np.delete(np.arange(self.nr_electodes), configs['input_indices'])
        self.nr_cv = len(self.indx_cv)
        offset = self.info['offset']
        amplitude = self.info['amplitude']

        self.min_voltage = offset - amplitude
        self.max_voltage = offset + amplitude
        bias = self.min_voltage[self.indx_cv] + \
            (self.max_voltage[self.indx_cv] - self.min_voltage[self.indx_cv]) * \
            np.random.rand(1, self.nr_cv)

        bias = TorchUtils.get_tensor_from_numpy(bias)
        self.bias = nn.Parameter(bias)
        # Set as torch Tensors and send to DEVICE
        self.indx_cv = TorchUtils.get_tensor_from_list(self.indx_cv, torch.int64)  # IndexError: tensors used as indices must be long, byte or bool tensors

        self.amplification = TorchUtils.get_tensor_from_list(self.info['amplification'])
        self.min_voltage = TorchUtils.get_tensor_from_list(self.min_voltage)
        self.max_voltage = TorchUtils.get_tensor_from_list(self.max_voltage)
        self.control_low = self.min_voltage[self.indx_cv]
        self.control_high = self.max_voltage[self.indx_cv]

    def get_output(self, input_matrix):
        with torch.no_grad():
            inputs_torch = TorchUtils.get_tensor_from_numpy(input_matrix)
            output = self.forward(inputs_torch)
        return TorchUtils.get_numpy_from_tensor(output)

    def forward(self, x):

        expand_cv = self.bias.expand(x.size()[0], -1)
        inp = torch.empty((x.size()[0], x.size()[1] + self.nr_cv))
        inp = TorchUtils.format_tensor(inp)
        inp[:, self.in_list] = x
        inp[:, self.indx_cv] = expand_cv

        return self.model(inp) * self.amplification

    def regularizer(self):
        assert any(self.control_low < 0), \
            "Min. Voltage is assumed to be negative, but value is positive!"
        assert any(self.control_high > 0), \
            "Max. Voltage is assumed to be positive, but value is negative!"
        return torch.sum(torch.relu(self.control_low - self.bias) + torch.relu(self.bias - self.control_high))

    def reset(self):
        for k in range(len(self.control_low)):
            print(f'    resetting control {k} between : {self.control_low[k], self.control_high[k]}')
            self.bias.data[:, k].uniform_(self.control_low[k], self.control_high[k])


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
