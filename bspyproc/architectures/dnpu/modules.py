"""
Created on Wed Jan 15 2020
Here you can find all classes defining the modules of DNPU architectures. All modules are child classes of TorchModel, 
which has nn.Module of PyTorch as parent.
@author: hruiz
"""


import torch
import numpy as np
import torch.nn as nn
from more_itertools import grouper
from bspyproc.processors.simulation.network import TorchModel
from bspyproc.utils.pytorch import TorchUtils


class DNPU_Base(TorchModel):
    '''DNPU Base class with activation nodes. All nodes are given by the same function loaded using the config dictionary configs_model.
    The argument inputs_list is a list containing the indices for the data inputs in each node. The length of this list defines the number
    of nodes and the elements of this list, are lists of integers. The number of inputs to the layer is defined by the total 
    number of integers in these lists.
    '''

    def __init__(self, inputs_list, configs_model):
        super().__init__(configs_model)

        ######### Set up node #########
        # Freeze parameters of node
        for params in self.parameters():
            params.requires_grad = False

        self.indices_node = np.arange(len(self.amplitude))
        ######### set learnable parameters #########
        control_list = self.set_controls(inputs_list)

        ###### Set everything as torch Tensors and send to DEVICE ######
        self.inputs_list = TorchUtils.get_tensor_from_list(inputs_list, torch.int64)
        self.control_list = TorchUtils.get_tensor_from_list(control_list, torch.int64)  # IndexError: tensors used as indices must be long, byte or bool tensors

    def set_controls(self, inputs_list):
        control_list = [np.delete(self.indices_node, indx) for indx in inputs_list]
        control_low = [self.min_voltage[indx_cv] for indx_cv in control_list]
        control_high = [self.max_voltage[indx_cv] for indx_cv in control_list]
        # Sample control parameters
        controls = [self.sample_controls(low, high) for low, high in zip(control_low, control_high)]
        # Register as learnable parameters
        self.all_controls = nn.ParameterList([nn.Parameter(cv) for cv in controls])
        # Set everything as torch Tensors and send to DEVICE
        self.control_low = torch.stack(control_low)
        self.control_high = torch.stack(control_high)
        return control_list

    def sample_controls(self, low, high):
        samples = TorchUtils.format_tensor(torch.rand(1, len(low)))
        return low + (high - low) * samples

    def evaluate_node(self, x, x_indices, controls, c_indices):
        expand_controls = controls.expand(x.size()[0], -1)
        data = torch.empty((x.size()[0], x.size()[1] + controls.size()[1]))
        data = TorchUtils.format_tensor(data)
        data[:, x_indices] = x
        data[:, c_indices] = expand_controls
        return self.model(data) * self.amplification

    def regularizer(self, beta=1.0):
        assert any(self.control_low < 0), \
            "Min. Voltage is assumed to be negative, but value is positive!"
        assert any(self.control_high > 0), \
            "Max. Voltage is assumed to be positive, but value is negative!"
        return beta * torch.sum(torch.relu(self.control_low - self.all_controls) + torch.relu(self.all_controls - self.control_high))

    def reset(self):
        raise NotImplementedError("Resetting controls not implemented!!")
        # for k in range(len(self.control_low)):
        #     # print(f'    resetting control {k} between : {self.control_low[k], self.control_high[k]}')
        #     self.controls.data[:, k].uniform_(self.control_low[k], self.control_high[k])


class DNPU_Layer(DNPU_Base):
    '''Layer with DNPUs as activation nodes. It is a child of the DNPU_base class with a forward method that implements the evaluation of this activation layer.
    The input data is partitioned into chunks of equal length assuming that this is the input dimension for each node. This partition is done by a generator method 
    self.partition_input(data).
    '''

    def __init__(self, inputs_list, configs_model):
        super().__init__(inputs_list, configs_model)

    def forward(self, x):
        assert x.shape[-1] == self.inputs_list.numel(), f'size mismatch: data is {x.shape}, DNPU_Layer expecting {self.inputs_list.numel()}'
        outputs = [self.evaluate_node(partition, self.inputs_list[i_node],
                                      self.all_controls[i_node], self.control_list[i_node])
                   for i_node, partition in enumerate(self.partition_input(x))]

        return torch.cat(outputs, dim=1)

    def partition_input(self, x):
        i = 0
        while i + self.inputs_list.shape[-1] <= x.shape[-1]:
            yield x[:, i:i + self.inputs_list.shape[-1]]
            i += self.inputs_list.shape[-1]


class DNPU_Channels(DNPU_Base):
    '''Layer with DNPU activation nodes expanding a small dimensional <7 input into a N-dimensional output where N is the number of nodes. It is a child of the DNPU_base class with a forward method that implements the evaluation of this activation layer.
    The input data to each node is assumed equal but it can be fed to each node differently. This is regulated with the list of input indices.
    '''

    def __init__(self, inputs_list, configs_model):
        super().__init__(inputs_list, configs_model)

    def forward(self, x):
        assert x.shape[-1] == len(self.inputs_list[0]), f'size mismatch: data is {x.shape}, DNPU_Channels expecting {len(self.inputs_list[0])}'
        outputs = [self.evaluate_node(x, self.inputs_list[i_node],
                                      self.all_controls[i_node], controls)
                   for i_node, controls in enumerate(self.control_list)]

        return torch.cat(outputs, dim=1)


if __name__ == '__main__':
    from bspyalgo.utils.io import load_configs
    import matplotlib.pyplot as plt
    import time

    NODE_CONFIGS = load_configs('configs/configs_nn_model.json')
    linear_layer = nn.Linear(20, 3).to(device=TorchUtils.get_accelerator_type())
    dnpu_layer = DNPU_Channels([[0, 3, 4]] * 1000, NODE_CONFIGS)

    model = nn.Sequential(linear_layer, dnpu_layer)

    data = torch.rand((200, 20)).to(device=TorchUtils.get_accelerator_type())
    start = time.time()
    output = model(data)
    end = time.time()

    print([param.shape for param in model.parameters() if param.requires_grad])
    print(f'(inputs,outputs) = {output.shape} of layer evaluated in {end-start} seconds')
    print(f'Output range : [{output.min()},{output.max()}]')

    plt.hist(output.flatten().cpu().detach().numpy(), bins=100)
    plt.show()
