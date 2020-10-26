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
        super().__init__()
        if isinstance(processor, Processor) or isinstance(processor, dict):
            self.base = DNPU_Base(processor, inputs_list) # It accepts initialising a processor as a dictionary
        else:
            self.base = processor # It accepts initialising as an external DNPU_Base
        self.window_size = window_size
        self.inputs_list = inputs_list
        self.out_size = out_size

    def forward(self, x):
        x = nf.unfold(x, kernel_size=self.window_size, stride=self.window_size)
        # x = (x[:, 1] * torch.tensor([2], dtype=torch.float32) + x[:, 0]) * (x[:, 2] * torch.tensor([2], dtype=torch.float32) + x[:, 3])
        x = torch.cat(
            [
                self.base(
                    x[:, :, i_node],
                    self.inputs_list[i_node],
                    self.base.all_controls[i_node],
                    self.base.control_list[i_node],
                )
                for i_node, controls in enumerate(self.control_list)
            ],
            dim=1,
        )
        if out_size is None:
            return x  
        else:
            return x.view(-1, self.out_size, self.out_size)
    
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
