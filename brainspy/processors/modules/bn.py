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
from brainspy.utils.transforms import CurrentToVoltage

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
        inputs_list=None, # It allows to declare a DNPU_Layer from a configs dictionary
        input_clip=True, # Whether if the input will be clipped by the 
        transform_to_voltage=True,
        input_range=None, # For example: [-1,1], this variable is required if there is an input clip or a transformation to voltage
        device_output_clip=True,
        batch_norm=True,
        bn_outputs=1,
        track_running_stats=True
    ):
        # default current_range = 2  * std, where std is assumed to be 1
        super().__init__()
        self.input_clip = input_clip
        self.device_output_clip = device_output_clip
        if input_range is None:
            assert input_clip is False and transform_to_voltage is False

        self.init_processor(processor, inputs_list)
        self.init_input_range(input_range)
        self.init_transform_to_voltage(transform_to_voltage, self.input_range)
        self.init_batch_norm(batch_norm, bn_outputs, track_running_stats)


    def init_processor(self, processor, inputs_list):
        if isinstance(processor, Processor) or isinstance(processor, dict):  # It accepts initialising a processor as a dictionary
            if inputs_list is None:
                self.processor = DNPU(processor)
            else:
                self.processor = DNPU_Layer(processor, inputs_list)
        elif isinstance(processor, DNPU) or isinstance(processor, DNPU_Layer):
            self.processor = processor
        else:
           assert False, 'The node is not recognised. It needs to be either a model dictionary or an instance of a Processor, a DNPU, or a DNPU_Layer.'

    def init_input_range(self, input_range):
        self.min_input = input_range[0]
        self.max_input = input_range[1]
        self.input_range = torch.ones_like(self.processor.get_input_ranges())
        self.input_range[:,0] *= self.min_input
        self.input_range[:,1] *= self.max_input

    def init_batch_norm(self,batch_norm, outputs, track_running_stats):
        if batch_norm:
            self.bn = nn.BatchNorm1d(outputs, affine=False, track_running_stats=track_running_stats).to(device=TorchUtils.get_accelerator_type())
        else:
            self.bn = batch_norm

    def init_transform_to_voltage(self, transform_to_voltage, input_range):
        if transform_to_voltage:
            self.transform_to_voltage = CurrentToVoltage(
                input_range, self.processor.get_input_ranges()
            )
        else:
            self.transform_to_voltage = transform_to_voltage

    def clamp_input(self,x):
        if self.input_clip:
            x = torch.clamp(x,min=self.min_input, max=self.max_input)
        return x

    def transform_input(self,x):
        if self.transform_to_voltage:
            x = self.transform_to_voltage(x)
        return x
    
    def clamp_output(self,x): 
        if self.device_output_clip:
            x = torch.clamp(x, min=self.processor.get_clipping_value()[0], max=self.processor.get_clipping_value()[1])
        return x

    def apply_batch_norm(self, x):
        if self.bn:
            x = self.bn(x)
        return x

    def forward(self, x):
        self.clamped_input = self.clamp_input(x)
        self.transformed_input = self.transform_input(self.clamped_input)
        self.dnpu_output = self.processor(self.transformed_input)
        self.clamped_dnpu_output = self.clamp_output(self.dnpu_output)
        self.batch_norm_output = self.apply_batch_norm(self.clamped_dnpu_output)
        
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
        return({'a_clamped_input':self.clamped_input,'b_transformed_input':self.transformed_input,'c_dnpu_output':self.dnpu_output,'d_clamped_dnpu_output':self.clamped_dnpu_output,'e_batch_norm_output':self.batch_norm_output})


if __name__ == "__main__":
    from brainspy.utils.io import load_configs
    import matplotlib.pyplot as plt
    import time

    NODE_CONFIGS = load_configs(
        "/home/hruiz/Documents/PROJECTS/DARWIN/Code/packages/brainspy/brainspy-processors/configs/configs_nn_model.json"
    )
    node = DNPU(NODE_CONFIGS)
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
    print(
        f"(inputs,outputs) = {output.shape} of layer evaluated in {end-start} seconds"
    )
    print(f"Output range : [{output.min()},{output.max()}]")

    plt.hist(output.flatten().cpu().detach().numpy(), bins=100)
    plt.show()
