"""
Created on Wed Jan 15 2020
Here you can find all classes defining the modules of DNPU architectures. All modules are child classes of TorchModel,
which has nn.Module of PyTorch as parent.
@author: hruiz
"""

import torch
import numpy as np
import torch.nn as nn
from brainspy.utils.pytorch import TorchUtils


class DNPU_Base(nn.Module):
    """DNPU Base class with activation nodes. All nodes are given by the same function loaded using the config dictionary configs_model.
    The argument inputs_list is a list containing the indices for the data inputs in each node. The length of this list defines the number
    of nodes and the elements of this list, are lists of integers. The number of inputs to the layer is defined by the total
    number of integers in these lists.
    """

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
        self.control_list = TorchUtils.get_tensor_from_list(
            control_list, torch.int64
        )  # IndexError: tensors used as indices must be long, byte or bool tensors

    def set_controls(self, inputs_list):
        control_list = [np.delete(self.indices_node, indx) for indx in inputs_list]
        control_low = [self.node.min_voltage[indx_cv] for indx_cv in control_list]
        control_high = [self.node.max_voltage[indx_cv] for indx_cv in control_list]
        # Sample control parameters
        controls = [
            self.sample_controls(low, high)
            for low, high in zip(control_low, control_high)
        ]
        # Register as learnable parameters
        self.all_controls = nn.ParameterList([nn.Parameter(cv) for cv in controls])
        # Set everything as torch Tensors and send to DEVICE
        self.control_low = torch.stack(control_low)
        self.control_high = torch.stack(control_high)
        return control_list

    def sample_controls(self, low, high):
        samples = torch.rand(1, len(low), device=TorchUtils.get_accelerator_type(), dtype=TorchUtils.get_data_type())
        return low + (high - low) * samples

    def evaluate_node(self, x, x_indices, controls, c_indices):
        expand_controls = controls.expand(x.size()[0], -1)
        data = torch.empty((x.size()[0], x.size()[1] + controls.size()[1]), device=TorchUtils.get_accelerator_type(), dtype=TorchUtils.get_data_type())
        data[:, x_indices] = x
        data[:, c_indices] = expand_controls
        return self.node(data) * self.node.amplification

    def regularizer(self):
        assert any(
            self.control_low.min(dim=0)[0] < 0
        ), "Min. Voltage is assumed to be negative, but value is positive!"
        assert any(
            self.control_high.max(dim=0)[0] > 0
        ), "Max. Voltage is assumed to be positive, but value is negative!"
        buff = 0.0
        for i, p in enumerate(self.all_controls):
            buff += torch.sum(
                torch.relu(self.control_low[i] - p)
                + torch.relu(p - self.control_high[i])
            )
        return buff

    def reset(self):
        raise NotImplementedError("Resetting controls not implemented!!")
        # for k in range(len(self.control_low)):
        #     # print(f'    resetting control {k} between : {self.control_low[k], self.control_high[k]}')
        #     self.controls.data[:, k].uniform_(self.control_low[k], self.control_high[k])
