import torch
import torch.nn as nn
from brainspy.processors.modules.base import DNPU_Base
from brainspy.processors.processor import Processor
import torch.nn.functional as nf


class Local_Receptive_Field(nn.Module):
    """Layer of DNPU nodes taking squared patches of images as inputs. The patch size is 2x2 so
    the number of inputs in the inputs_list elements must be 4. The pathes are non-overlapping.
    """

    def __init__(self, processor, inputs_list, out_size=None, window_size=2):
        super(Local_Receptive_Field, self).__init__()
        if isinstance(processor, Processor) or isinstance(processor, dict):
            self.processor = DNPU_Base(processor, inputs_list) # It accepts initialising a processor as a dictionary
        else:
            self.processor = processor # It accepts initialising as an external DNPU_Base
        self.window_size = window_size
        self.inputs_list = inputs_list
        self.out_size = out_size

    def forward(self, x):
        x = nf.unfold(x, kernel_size=self.window_size, stride=self.window_size)
        # x = (x[:, 1] * torch.tensor([2], dtype=torch.float32) + x[:, 0]) * (x[:, 2] * torch.tensor([2], dtype=torch.float32) + x[:, 3])
        x = torch.cat(
            [
                self.processor(
                    x[:, :, i_node],
                    self.inputs_list[i_node],
                    self.processor.all_controls[i_node],
                    self.processor.control_list[i_node],
                )
                for i_node, controls in enumerate(self.processor.control_list)
            ],
            dim=1,
        )
        if self.out_size is None:
            return x  
        else:
            return x.view(-1, self.out_size, self.out_size)
    
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
