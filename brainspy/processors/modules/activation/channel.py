import torch
from brainspy.processors.modules.activation.base import DNPU_Base


class DNPU_Channels(DNPU_Base):
    """Layer with DNPU activation nodes expanding a small dimensional <7 input
    into a N-dimensional output where N is the number of nodes.
    It is a child of the DNPU_base class that implements the evaluation of this
    activation layer using the model provided.
    The input data to each node is assumed equal but it can be fed to each node
    differently. This is regulated with the list of input indices.
    """

    def __init__(self, model, inputs_list):
        super().__init__(inputs_list, model)

    def forward(self, x):
        assert x.shape[-1] == len(
            self.inputs_list[0]
        ), f"size mismatch: data is {x.shape}, DNPU_Channels expecting {len(self.inputs_list[0])}"
        outputs = [
            self.evaluate_node(
                x, self.inputs_list[i_node], self.all_controls[i_node], controls
            )
            for i_node, controls in enumerate(self.control_list)
        ]

        return torch.cat(outputs, dim=1)
