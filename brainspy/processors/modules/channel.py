import torch
import torch.nn as nn
from brainspy.processors.modules.base import DNPU_Base
from brainspy.processors.processor import Processor

class DNPU_Channels(nn.Module):
    """Layer with DNPU activation nodes expanding a small dimensional <7 input
    into a N-dimensional output where N is the number of nodes.
    It is a child of the DNPU_base class that implements the evaluation of this
    activation layer using the model provided.
    The input data to each node is assumed equal but it can be fed to each node
    differently. This is regulated with the list of input indices.
    """

    def __init__(self, processor, inputs_list):
        super().__init__()
        if isinstance(processor, Processor) or isinstance(processor, dict):
            self.base = DNPU_Base(processor, inputs_list) # It accepts initialising a processor as a dictionary
        else:
            self.base = processor # It accepts initialising as an external DNPU_Base

    def forward(self, x):
        assert x.shape[-1] == len(
            self.base.inputs_list[0]
        ), f"size mismatch: data is {x.shape}, DNPU_Channels expecting {len(self.base.inputs_list[0])}"
        outputs = [
            self.base(
                x, self.base.inputs_list[i_node], self.base.all_controls[i_node], controls
            )
            for i_node, controls in enumerate(self.base.control_list)
        ]

        return torch.cat(outputs, dim=1)

    def regularizer(self):
        return self.base.regularizer()

    def hw_eval(self, hw_processor_configs):
        self.base.hw_eval(hw_processor_configs)

    def is_hardware(self):
        return self.base.is_hardware()

    def get_clipping_value(self):
        return self.base.get_clipping_value()

    def get_control_ranges(self):
        return self.base.get_control_ranges()

    def get_control_voltages(self):
        return self.base.get_control_voltages()

    def set_control_voltages(self, control_voltages):
        return self.base.set_control_voltages(control_voltages)