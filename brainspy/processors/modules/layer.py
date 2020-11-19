import torch
import torch.nn as nn
from brainspy.processors.modules.base import DNPU_Base
from brainspy.processors.processor import Processor

class DNPU_Layer(nn.Module):
    """Layer with DNPUs as activation nodes. It is a child of the DNPU_base class that implements
    the evaluation of this activation layer given by the model provided.
    The input data is partitioned into chunks of equal length assuming that this is the
    input dimension for each node. This partition is done by a generator method
    self.partition_input(data).
    """

    def __init__(self, processor, inputs_list):
        super(DNPU_Layer, self).__init__()
        if isinstance(processor, Processor) or isinstance(processor, dict):
            self.processor = DNPU_Base(processor, inputs_list)  # It accepts initialising a processor as a dictionary
        else:
            self.processor = processor  # It accepts initialising as an external DNPU_Base

    def forward(self, x):
        assert (
            x.shape[-1] == self.processor.inputs_list.numel()
        ), f"size mismatch: data is {x.shape}, DNPU_Layer expecting {self.processor.inputs_list.numel()}"
        outputs = [
            self.processor(
                partition,
                self.processor.inputs_list[i_node],
                self.processor.all_controls[i_node],
                self.processor.control_list[i_node],
            )
            for i_node, partition in enumerate(self.partition_input(x))
        ]

        return torch.cat(outputs, dim=1)

    def partition_input(self, x):
        i = 0
        while i + self.processor.inputs_list.shape[-1] <= x.shape[-1]:
            yield x[:, i: i + self.processor.inputs_list.shape[-1]]
            i += self.processor.inputs_list.shape[-1]

    def regularizer(self):
        return self.processor.regularizer()

    def hw_eval(self, hw_processor_configs):
        self.processor.hw_eval(hw_processor_configs)

    def is_hardware(self):
        return self.processor.is_hardware()

    def get_clipping_value(self):
        return self.processor.get_clipping_value()

    def get_input_ranges(self):
        return self.processor.get_input_ranges()

    def get_control_ranges(self):
        return self.processor.get_control_ranges()

    def get_control_voltages(self):
        return self.processor.get_control_voltages()

    def set_control_voltages(self, control_voltages):
        return self.processor.set_control_voltages(control_voltages)
