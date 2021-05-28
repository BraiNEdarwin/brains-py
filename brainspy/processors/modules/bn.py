"""
Created on Wed Jan 15 2020
Here you can find all classes defining the modules of DNPU architectures. All modules are child classes of TorchModel,
which has nn.Module of PyTorch as parent.
@author: hruiz
"""


import torch

import torch.nn as nn

from brainspy.processors.dnpu import DNPU
from brainspy.utils.pytorch import TorchUtils
from brainspy.utils.transforms import get_linear_transform_constants

from brainspy.processors.modules.layer import DNPU_Layer
from brainspy.processors.processor import Processor


class DNPU_BatchNorm(nn.Module):
    """
    v_min value is the minimum voltage value of the electrode of the next dnpu to which the output is going to be connected
    v_max value is the maximum voltage value of the electrode of the next dnpu to which the output is going to be connected
    """

    def __init__(
        self,
        processor,  # It is either a dictionary or the reference to a processor or a DNPU_Base module, or any other of the modules channel, layer, lrf
        inputs_list=None,  # It allows to declare a DNPU_Layer from a configs dictionary
        dnpu_type='for',
        # Transformation configs
        # input_clip=True,  # Whether if the input will be clipped by the
        # transform_to_voltage=True,  # Whether if a transformation to from the data input range to the data input voltages will be applied.
        # input_range=None,  # For example: [-1,1], this variable is required if there is an input clip or a transformation to voltage
        # # Output clipping  configs
        # device_output_clip=True,  # Whether the device output will be clipped with the same clipping values as the setup used for gathering the training data with which it was trained
        # # Batch norm related configs
        # batch_norm=True,  # Whether if batch norm is applied
        # affine=False,
        # track_running_stats=True,  # Whether the batchnorm will track the running stats.
        # momentum=0.1,
    ):
        # default current_range = 2  * std, where std is assumed to be 1
        super(DNPU_BatchNorm, self).__init__()
        # self.input_clip = input_clip
        # self.device_output_clip = device_output_clip
        self.init_processor(processor, inputs_list, dnpu_type)

        self.input_clip = False
        self.transform_to_voltage = False
        self.bn = False
        # else:
            
        #     self.init_transform_to_voltage(transform_to_voltage, self.input_range)
        # self.init_batch_norm(batch_norm, affine, track_running_stats, momentum)

    def init_processor(self, processor, inputs_list, dnpu_type):
        self.processor = DNPU(processor, inputs_list, dnpu_type)

    def init_input_range(self, input_range):
        self.min_input = input_range[0]
        self.max_input = input_range[1]
        input_range = torch.ones_like(self.processor.get_input_ranges()).T
        
        input_range[0] *= self.min_input
        input_range[1] *= self.max_input
        self.input_range = input_range.T

    def init_batch_norm(self, affine=False, track_running_stats=True, momentum=0.1, eps=1e-5, custom_bn=None):
        if custom_bn is None:
            self.bn = nn.BatchNorm1d(
                self.processor.get_node_no(),
                affine=affine,
                track_running_stats=track_running_stats,
                momentum=momentum,
                eps=eps
            ).to(device=TorchUtils.get_device())
        else:
            self.bn = custom_bn

    # def init_output_node_no(self):
    #     if isinstance(self.processor, DNPU):
    #         self.bn_outputs = 1  # Number of outputs from the DNPU layer
    #     elif isinstance(self.processor, DNPU_Layer):
    #         self.bn_outputs = len(self.processor.processor.inputs_list)
    #     else:
    #         print(
    #             "Warning: self.processor in DNPU_BatchNorm is from a type that is not identified. The outputs of batch norm are not automatically detected."
    #         )

    # Strict defines if the input is going to be clipped before doing the linear transformation in order to ensure that the transformation is correct
    def init_transform_to_voltage(self, input_range, strict=True):
        self.input_clip = strict
        self.init_input_range(input_range)
        self.transform_to_voltage = True
        # self.transform_to_voltage = CurrentToVoltage(
        #     self.input_range, self.processor.get_input_ranges()
        # )
        scale, offset = get_linear_transform_constants(self.processor.get_input_ranges().T[0].T, self.processor.get_input_ranges().T[1].T, self.input_range.T[0].T, self.input_range.T[1].T)
        self.scale = scale.flatten()
        self.offset = offset.flatten()

    def clamp_input(self, x):
        if self.input_clip:
            x = torch.clamp(x, min=self.min_input, max=self.max_input)
        return x

    def transform_input(self, x):
        if self.transform_to_voltage:
            x = (self.scale * x) + self.offset  # self.transform_to_voltage(x)
        return x

    def apply_batch_norm(self, x):
        if self.bn:
            x = self.bn(x)
        return x

    def forward(self, x):
        self.clamped_input = self.clamp_input(x)
        self.transformed_input = self.transform_input(self.clamped_input)
        self.dnpu_output = self.processor(self.transformed_input)
        self.batch_norm_output = self.apply_batch_norm(self.dnpu_output)

        return self.batch_norm_output

    def regularizer(self):
        return self.processor.regularizer()

    def hw_eval(self, hw_processor_configs):
        self.processor.hw_eval(hw_processor_configs)

    def is_hardware(self):
        return self.processor.is_hardware()

    def get_clipping_value(self):
        return self.processor.get_clipping_value()

    def get_control_ranges(self):
        return self.processor.get_control_ranges()

    def get_control_voltages(self):
        return self.processor.get_control_voltages()

    def set_control_voltages(self, control_voltages):
        return self.processor.set_control_voltages(control_voltages)

    def get_logged_variables(self):
        return {
            "a_clamped_input": self.clamped_input.clone().detach(),
            "b_transformed_input": self.transformed_input.clone().detach(),
            "c_dnpu_output": self.dnpu_output.clone().detach(),
            "d_batch_norm_output": self.batch_norm_output.clone().detach(),
        }