'''Author: HC Ruiz Euler and Unai Alegre Ibarra; 
DNPU based network of devices to solve complex tasks 25/10/2019

'''

import torch
import numpy as np
import torch.nn as nn
from bspyproc.utils.pytorch import TorchUtils
from bspyproc.processors.simulation.dopanet import DNPU
from bspyproc.architectures.architectures import DNPUArchitecture


class TwoToOneDNPU(DNPUArchitecture):

    def __init__(self, configs):  # in_dict, path=r'../Data/Models/checkpoint3000_02-07-23h47m.pt'
        super().__init__(configs)
        # self.in_dict = in_dict
        # self.conversion_offset = torch.tensor(-0.6)

        self.input_node1 = DNPU(in_dict['input_node1'], path=path)
        self.input_node2 = DNPU(in_dict['input_node2'], path=path)
        self.bn1 = nn.BatchNorm1d(2, affine=False)
        self.output_node = DNPU(in_dict['output_node'], path=path)

    def forward(self, x):
        # Pass through input layer
        x = (self.scale * x) + self.offset
        x1 = self.input_node1(x)
        x2 = self.input_node2(x)
        # --- BatchNorm --- #
        h = self.bn1(torch.cat((x1, x2), dim=1))
        std1 = np.sqrt(torch.mean(self.bn1.running_var).cpu().numpy())
        cut = 2 * std1
        # Pass it through output layer
        h = torch.tensor(1.8 / (4 * std1)) * \
            torch.clamp(h, min=-cut, max=cut) + self.conversion_offset
        return self.output_node(h)

    def regularizer(self):
        control_penalty = self.input_node1.regularizer() \
            + self.input_node2.regularizer() \
            + self.output_node.regularizer()
        return control_penalty + self.offset_penalty() + self.scale_penalty()
