"""
Created on Wed Jan 15 2020
Here you can find all classes defining the modules of DNPU architectures. All modules are child classes of TorchModel,
which has nn.Module of PyTorch as parent.
@author: hruiz
"""


import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as nf
from more_itertools import grouper
from bspyproc.processors.simulation.network import TorchModel
from bspyproc.utils.pytorch import TorchUtils


class DNPU_Base(nn.Module):
    '''DNPU Base class with activation nodes. All nodes are given by the same function loaded using the config dictionary configs_model.
    The argument inputs_list is a list containing the indices for the data inputs in each node. The length of this list defines the number
    of nodes and the elements of this list, are lists of integers. The number of inputs to the layer is defined by the total
    number of integers in these lists.
    '''

    def __init__(self, inputs_list, model):
        super().__init__()
        self.node = model
        ######### Set up node #########
        # Freeze parameters of node
        for params in self.node.parameters():
            params.requires_grad = False

        self.indices_node = np.arange(len(self.node.amplitude))
        ######### set learnable parameters #########
        control_list = self.set_controls(inputs_list)

        ###### Set everything as torch Tensors and send to DEVICE ######
        self.inputs_list = TorchUtils.get_tensor_from_list(inputs_list, torch.int64)
        self.control_list = TorchUtils.get_tensor_from_list(control_list, torch.int64)  # IndexError: tensors used as indices must be long, byte or bool tensors

    def set_controls(self, inputs_list):
        control_list = [np.delete(self.indices_node, indx) for indx in inputs_list]
        control_low = [self.node.min_voltage[indx_cv] for indx_cv in control_list]
        control_high = [self.node.max_voltage[indx_cv] for indx_cv in control_list]
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
        return self.node(data) * self.node.amplification

    def regularizer(self):
        assert any(self.control_low.min(dim=0)[0] < 0), \
            "Min. Voltage is assumed to be negative, but value is positive!"
        assert any(self.control_high.max(dim=0)[0] > 0), \
            "Max. Voltage is assumed to be positive, but value is negative!"
        buff = 0.
        for i, p in enumerate(self.all_controls):
            buff += torch.sum(torch.relu(self.control_low[i] - p)
                              + torch.relu(p - self.control_high[i]))
        return buff

    def reset(self):
        raise NotImplementedError("Resetting controls not implemented!!")
        # for k in range(len(self.control_low)):
        #     # print(f'    resetting control {k} between : {self.control_low[k], self.control_high[k]}')
        #     self.controls.data[:, k].uniform_(self.control_low[k], self.control_high[k])


class DNPU_Layer(DNPU_Base):
    '''Layer with DNPUs as activation nodes. It is a child of the DNPU_base class that implements
    the evaluation of this activation layer given by the model provided.
    The input data is partitioned into chunks of equal length assuming that this is the
    input dimension for each node. This partition is done by a generator method
    self.partition_input(data).
    '''

    def __init__(self, model, inputs_list):
        super().__init__(inputs_list, model)

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
    '''Layer with DNPU activation nodes expanding a small dimensional <7 input
    into a N-dimensional output where N is the number of nodes.
    It is a child of the DNPU_base class that implements the evaluation of this
    activation layer using the model provided.
    The input data to each node is assumed equal but it can be fed to each node
    differently. This is regulated with the list of input indices.
    '''

    def __init__(self, model, inputs_list):
        super().__init__(inputs_list, model)

    def forward(self, x):
        assert x.shape[-1] == len(self.inputs_list[0]), f'size mismatch: data is {x.shape}, DNPU_Channels expecting {len(self.inputs_list[0])}'
        outputs = [self.evaluate_node(x, self.inputs_list[i_node],
                                      self.all_controls[i_node], controls)
                   for i_node, controls in enumerate(self.control_list)]

        return torch.cat(outputs, dim=1)


class Local_Receptive_Field(DNPU_Base):
    '''Layer of DNPU nodes taking squared patches of images as inputs. The patch size is 2x2 so
     the number of inputs in the inputs_list elements must be 4. The pathes are non-overlapping.
    '''

    def __init__(self, model, inputs_list, out_size):
        super().__init__(inputs_list, model)
        self.window_size = 2
        self.inputs_list = inputs_list
        self.out_size = out_size

    def forward(self, x):
        x = nf.unfold(x, kernel_size=self.window_size, stride=self.window_size)
        # x = (x[:, 1] * torch.tensor([2], dtype=torch.float32) + x[:, 0]) * (x[:, 2] * torch.tensor([2], dtype=torch.float32) + x[:, 3])
        x = torch.cat([self.evaluate_node(x[:, :, i_node], self.inputs_list[i_node],
                                          self.all_controls[i_node], self.control_list[i_node])
                       for i_node, controls in enumerate(self.control_list)], dim=1)
        return x.view(-1, self.out_size, self.out_size)


if __name__ == '__main__':
    from bspyalgo.utils.io import load_configs
    import matplotlib.pyplot as plt
    import time

    NODE_CONFIGS = load_configs('/home/hruiz/Documents/PROJECTS/DARWIN/Code/packages/brainspy/brainspy-processors/configs/configs_nn_model.json')
    node = TorchModel(NODE_CONFIGS)
    # linear_layer = nn.Linear(20, 3).to(device=TorchUtils.get_accelerator_type())
    # dnpu_layer = DNPU_Channels([[0, 3, 4]] * 1000, node)
    linear_layer = nn.Linear(20, 300).to(device=TorchUtils.get_accelerator_type())
    dnpu_layer = DNPU_Layer([[0, 3, 4]] * 100, node)

    model = nn.Sequential(linear_layer, dnpu_layer)

    data = torch.rand((200, 20)).to(device=TorchUtils.get_accelerator_type())
    start = time.time()
    output = model(data)
    end = time.time()

    # print([param.shape for param in model.parameters() if param.requires_grad])
    print(f'(inputs,outputs) = {output.shape} of layer evaluated in {end-start} seconds')
    print(f'Output range : [{output.min()},{output.max()}]')

    plt.hist(output.flatten().cpu().detach().numpy(), bins=100)
    plt.show()
