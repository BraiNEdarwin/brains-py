import torch
from brainspy.processors.modules.activation.base import DNPU_Base


class DNPU_Layer(DNPU_Base):
    """Layer with DNPUs as activation nodes. It is a child of the DNPU_base class that implements
    the evaluation of this activation layer given by the model provided.
    The input data is partitioned into chunks of equal length assuming that this is the
    input dimension for each node. This partition is done by a generator method
    self.partition_input(data).
    """

    def __init__(self, model, inputs_list):
        super().__init__(inputs_list, model)

    def forward(self, x):
        assert (
            x.shape[-1] == self.inputs_list.numel()
        ), f"size mismatch: data is {x.shape}, DNPU_Layer expecting {self.inputs_list.numel()}"
        outputs = [
            self.evaluate_node(
                partition,
                self.inputs_list[i_node],
                self.all_controls[i_node],
                self.control_list[i_node],
            )
            for i_node, partition in enumerate(self.partition_input(x))
        ]

        return torch.cat(outputs, dim=1)

    def partition_input(self, x):
        i = 0
        while i + self.inputs_list.shape[-1] <= x.shape[-1]:
            yield x[:, i: i + self.inputs_list.shape[-1]]
            i += self.inputs_list.shape[-1]
