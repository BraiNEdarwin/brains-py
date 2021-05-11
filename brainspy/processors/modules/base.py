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
from brainspy.processors.processor import Processor


class DNPU_Base(nn.Module):
    """DNPU Base class with activation nodes. All nodes are given by the same function loaded using the config dictionary configs_model.
    The argument inputs_list is a list containing the indices for the data inputs in each node. The length of this list defines the number
    of nodes and the elements of this list, are lists of integers. The number of inputs to the layer is defined by the total
    number of integers in these lists.
    """

    def __init__(self, processor, inputs_list):
        super(DNPU_Base, self).__init__()
        if isinstance(processor, Processor):
            self.processor = processor
        else:
            self.processor = Processor(
                processor
            )  # It accepts initialising a processor as a dictionary
        # ######## Set up node #########
        # Freeze parameters of node
        for params in self.processor.parameters():
            params.requires_grad = False

        self.indices_node = np.arange(self.processor.get_activation_electrode_no())
        # ######## set learnable parameters #########
        self.control_list = TorchUtils.format(
            self.set_controls(inputs_list), data_type=torch.int64
        )

        # ######## Initialise data input ranges #########
        self.data_input_low = torch.stack(
            [
                self.processor.processor.voltage_ranges[indx_cv, 0]
                for indx_cv in inputs_list
            ]
        )
        self.data_input_high = torch.stack(
            [
                self.processor.processor.voltage_ranges[indx_cv, 1]
                for indx_cv in inputs_list
            ]
        )

        # ##### Set everything as torch Tensors and send to DEVICE ######
        self.inputs_list = TorchUtils.format(inputs_list, data_type=torch.int64)
        # IndexError: tensors used as indices must be long, byte or bool tensors

    def set_controls(self, inputs_list):
        control_list = [np.delete(self.indices_node, indx) for indx in inputs_list]
        control_low = [
            self.processor.processor.voltage_ranges[indx_cv, 0]
            for indx_cv in control_list
        ]
        control_high = [
            self.processor.processor.voltage_ranges[indx_cv, 1]
            for indx_cv in control_list
        ]
        # Sample control parameters
        controls = [
            self.sample_controls(low, high)
            for low, high in zip(control_low, control_high)
        ]
        # Register as learnable parameters
        self.all_controls = nn.ParameterList(
            [nn.Parameter(cv) for cv in controls]
        )  # Throwing warning reported as bug at https://github.com/pytorch/pytorch/issues/46983
        # Set everything as torch Tensors and send to DEVICE

        self.control_low = torch.stack(control_low)
        self.control_high = torch.stack(control_high)

        return control_list

    def sample_controls(self, low, high):
        samples = torch.rand(
            1, len(low), device=TorchUtils.get_device(), dtype=torch.get_default_dtype()
        )
        return low + (high - low) * samples

    # Evaluate node
    def forward(self, x, x_indices, controls, c_indices):
        assert (
            x.dtype == controls.dtype and x.device == controls.device
        ), "Data types or devices not matching. "
        expand_controls = controls.expand(x.size()[0], -1)
        data = torch.empty(
            (x.size()[0], x.size()[1] + controls.size()[1]),
            device=x.device,
            dtype=x.dtype,
        )
        data[:, x_indices] = x
        data[:, c_indices] = expand_controls
        return self.processor.processor(data)  # * self.node.amplification

    def regularizer(self):
        if "control_low" not in dir(self) and "control_high" not in dir(self):
            return 0
        else:
            assert any(
                self.control_low.min(dim=0)[0] < 0
            ), "Min. Voltage is assumed to be negative, but value is positive!"
            assert any(
                self.control_high.max(dim=0)[0] > 0
            ), "Max. Voltage is assumed to be positive, but value is negative!"
            buff = 0.0
            for i, p in enumerate(self.all_controls):
                buff += torch.sum(
                    torch.relu(self.control_low[i] - p) + torch.relu(p - self.control_high[i])
                )
            return buff

    def hw_eval(self, arg):
        self.eval()
        if isinstance(arg, Processor):
            self.processor = arg
        else:
            self.processor.load_processor(arg)
        assert torch.equal(
            self.control_low.cpu().half(), self.get_control_ranges()[0, :].cpu().half()
        ), "Low control voltage ranges for the new processor are different than the control voltage ranges for which the DNPU was trained."
        assert torch.equal(
            self.control_high.cpu().half(), self.get_control_ranges()[1, :].cpu().half()
        ), "High control voltage ranges for the new processor are different than the control voltage ranges for which the DNPU was trained."

    def is_hardware(self):
        return self.processor.is_hardware

    def get_clipping_value(self):
        return self.processor.get_clipping_value()

    def get_input_ranges(self):
        return torch.cat(
            (
                self.data_input_low.flatten().unsqueeze(1),
                self.data_input_high.flatten().unsqueeze(1),
            ),
            dim=1,
        )

    def get_control_ranges(self):
        return torch.cat(
            (self.control_low.unsqueeze(0), self.control_high.unsqueeze(0)), dim=0
        )  # Total Dimensions 3: Dim 0: 0=min volt range1=max volt range, Dim 1: Index of node, Dim 2: Index of electrode

    def get_control_voltages(self):
        return torch.vstack([cv.data.detach() for cv in self.all_controls]).flatten()

    def set_control_voltages(self, control_voltages):
        with torch.no_grad():
            # bias = bias.unsqueeze(dim=0)
            assert (
                self.all_controls.shape == control_voltages.shape
            ), "Control voltages could not be set due to a shape missmatch with regard to the ones already in the model."
            self.bias = torch.nn.Parameter(TorchUtils.format(control_voltages))
