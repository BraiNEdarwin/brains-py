import torch
from brainspy.processors.modules.activation.base import DNPU_Base
import torch.nn.functional as nf


class Local_Receptive_Field(DNPU_Base):
    """Layer of DNPU nodes taking squared patches of images as inputs. The patch size is 2x2 so
    the number of inputs in the inputs_list elements must be 4. The pathes are non-overlapping.
    """

    def __init__(self, model, inputs_list, out_size):
        super().__init__(inputs_list, model)
        self.window_size = 2
        self.inputs_list = inputs_list
        self.out_size = out_size

    def forward(self, x):
        x = nf.unfold(x, kernel_size=self.window_size, stride=self.window_size)
        # x = (x[:, 1] * torch.tensor([2], dtype=torch.float32) + x[:, 0]) * (x[:, 2] * torch.tensor([2], dtype=torch.float32) + x[:, 3])
        x = torch.cat(
            [
                self.evaluate_node(
                    x[:, :, i_node],
                    self.inputs_list[i_node],
                    self.all_controls[i_node],
                    self.control_list[i_node],
                )
                for i_node, controls in enumerate(self.control_list)
            ],
            dim=1,
        )
        return x.view(-1, self.out_size, self.out_size)
