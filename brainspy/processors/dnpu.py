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
from brainspy.utils.electrodes import merge_electrode_data

from brainspy.processors.simulation.surrogate import SurrogateModel
from brainspy.processors.hardware.processor import HardwareProcessor


class DNPU(nn.Module):
    """"""

    def __init__(self, configs):
        super().__init__()
        self._load_processor(configs)
        self._init_electrode_info(configs)
        self._init_regularization_factor(configs)

        for (
            params
        ) in (
            self.parameters()
        ):  # Freeze parameters of the neural network of the surrogate model
            params.requires_grad = False
        self._init_bias()

    def _load_processor(self, configs):
        if not hasattr(self, 'processor') or self._get_configs() != configs:
            if configs["processor_type"] == "simulation":
                self.processor = SurrogateModel(configs)
                self.electrode_no = len(
                    self.processor.info["data_info"]["input_data"]["offset"]
                )
            elif configs["processor_type"] == "simulation_debug" or configs["processor_type"] == "cdaq_to_cdaq" or configs["processor_type"] == "cdaq_to_nidaq":
                self.processor = HardwareProcessor(configs)
                self.electrode_no = configs['data']['activation_electrode_no']
            else:
                raise NotImplementedError(
                    f"Platform {configs['platform']} is not recognised. The platform has to be either simulation, simulation_debug, cdaq_to_cdaq or cdaq_to_nidaq. "
                )

    def _init_electrode_info(self, configs):
        # self.input_no = len(configs['data_input_indices'])
        self.data_input_indices = TorchUtils.get_tensor_from_list(
            configs["data"]["input_indices"], data_type=torch.int64
        )
        self.control_indices = np.delete(
            np.arange(self.electrode_no), configs["data"]["input_indices"]
        )
        self.control_indices = TorchUtils.get_tensor_from_list(
            self.control_indices, data_type=torch.int64
        )  # IndexError: tensors used as indices must be long, byte or bool tensors

    def _init_regularization_factor(self, configs):
        if "regularisation_factor" in configs:
            self.alpha = torch.tensor(configs["regularisation_factor"], device=TorchUtils.get_accelerator_type(), dtype=TorchUtils.get_data_type())
        else:
            # print('No regularisation factor set.')
            self.alpha = torch.tensor([1], device=TorchUtils.get_accelerator_type(), dtype=TorchUtils.get_data_type())

    def _init_bias(self):
        self.control_low = self.processor.voltage_ranges[self.control_indices][:, 0]
        self.control_high = self.processor.voltage_ranges[self.control_indices][:, 1]
        assert any(
            self.control_low < 0
        ), "Min. Voltage is assumed to be negative, but value is positive!"
        assert any(
            self.control_high > 0
        ), "Max. Voltage is assumed to be positive, but value is negative!"
        bias = self.control_low + (
            self.control_high - self.control_low
        ) * TorchUtils.get_tensor_from_numpy(
            np.random.rand(1, len(self.control_indices))
        )

        self.bias = nn.Parameter(bias)

    def forward(self, x):
        inp = merge_electrode_data(
            x,
            self.bias.expand(x.size()[0], -1),
            self.data_input_indices,
            self.control_indices,
        )
        return self.processor(inp)

    def regularizer(self):
        return self.alpha * (
            torch.sum(
                torch.relu(self.control_low - self.bias)
                + torch.relu(self.bias - self.control_high)
            )
        )

    def reset(self):
        for k in range(len(self.control_low)):
            # print(f'    resetting control {k} between : {self.control_low[k], self.control_high[k]}')
            self.bias.data[:, k].uniform_(self.control_low[k], self.control_high[k])

    def get_input_ranges(self):
        return self.processor.voltage_ranges[self.data_input_indices]

    def get_control_ranges(self):
        return self.processor.voltage_ranges[self.control_indices]

    def set_control_voltages(self, bias):
        with torch.no_grad():
            bias = bias.unsqueeze(dim=0)
            assert (
                self.bias.shape == bias.shape
            ), "Control voltages could not be set due to a shape missmatch with regard to the ones already in the model."
            self.bias = nn.Parameter(TorchUtils.format_tensor(bias))

    def get_control_voltages(self):
        return next(self.parameters()).detach()

    def get_clipping_value(self):
        return self.processor.clipping_value

    def hw_eval(self, hw_processor_configs):
        self.eval()
        self._load_processor(hw_processor_configs)
        self._init_electrode_info(hw_processor_configs)

    def _get_configs(self):
        if isinstance(self.processor, HardwareProcessor):
            return self.processor.driver.configs
        elif isinstance(self.processor, SurrogateModel):
            return self.processor.configs
        else:
            print('Warning: Instance of processor not recognised.')
            return None

    def close(self):
        self.processor.close()

    def is_hardware(self):
        return self.processor.is_hardware()
