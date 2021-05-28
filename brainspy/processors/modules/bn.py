import torch.nn as nn

from brainspy.processors.dnpu import DNPU
from brainspy.utils.pytorch import TorchUtils
from brainspy.processors.processor import Processor


class DNPUBatchNorm(DNPU):
    """
    v_min value is the minimum voltage value of the electrode of the next dnpu to which the output is going to be connected
    v_max value is the maximum voltage value of the electrode of the next dnpu to which the output is going to be connected
    """

    def __init__(
        self,
        processor: Processor,
        data_input_indices: list,  # Data input electrode indices. It should be a double list (e.g., [[1,2]] or [[1,2],[1,3]])
        forward_pass_type: str = 'vec',  # It can be 'for' in order to do time multiplexing with the same device using a for loop, and 'vec' in order to do time multiplexing with vectorisation. )
        # Parameters related to BatchNorm1d from pytorch
        affine=False,
        track_running_stats=True,
        momentum=0.1,
        eps=1e-5,
        custom_bn=nn.BatchNorm1d
    ):
        super(DNPUBatchNorm, self).__init__(processor, data_input_indices, forward_pass_type=forward_pass_type)
        self.bn = custom_bn(
            self.get_node_no(),
            affine=affine,
            track_running_stats=track_running_stats,
            momentum=momentum,
            eps=eps
        ).to(device=TorchUtils.get_device())

    def forward(self, x):
        self.dnpu_output = self.forward_pass(x)
        self.batch_norm_output = self.bn(self.dnpu_output)
        return self.batch_norm_output

    def get_logged_variables(self):
        return {
            "c_dnpu_output": self.dnpu_output.clone().detach(),
            "d_batch_norm_output": self.batch_norm_output.clone().detach(),
        }