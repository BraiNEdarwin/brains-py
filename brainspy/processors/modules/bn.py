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


class DNPU_BatchNorm(nn.Module):
    """
    v_min value is the minimum voltage value of the electrode of the next dnpu to which the output is going to be connected
    v_max value is the maximum voltage value of the electrode of the next dnpu to which the output is going to be connected
    """

    def __init__(
        self,
        processor,  # It is either a dictionary or the reference to a processor
        current_range=torch.tensor([[-2, 2], [-2, 2]]),
        current_to_voltage=True,
        batch_norm=True,
    ):
        # default current_range = 2  * std, where std is assumed to be 1
        super().__init__()

        self.dnpu = DNPU(processor)  # DNPU(configs)

        if batch_norm:
            self.bn = nn.BatchNorm1d(1, affine=False).to(device=TorchUtils.get_accelerator_type())
        else:
            self.bn = batch_norm
        if current_to_voltage:
            self.current_to_voltage = CurrentToVoltage(
                current_range, self.dnpu.processor.get_input_ranges()
            )
        else:
            self.current_to_voltage = current_to_voltage

    def forward(self, x):
        if self.current_to_voltage:
            x = self.current_to_voltage(x)
        x = self.dnpu(x)
        # Cut off values out of the clipping value
        x = torch.clamp(
            x,
            min=self.dnpu.processor.get_clipping_value()[0],
            max=self.dnpu.processor.get_clipping_value()[1],
        )
        # Apply batch normalisation
        if self.bn:
            x = self.bn(x)
        # Apply current to voltage transformation
        # x = self.current_to_voltage(x)
        return x

    def hw_eval(self, hw_processor_configs):
        self.dnpu.hw_eval(hw_processor_configs)


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
