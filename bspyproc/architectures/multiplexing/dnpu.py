'''Author: HC Ruiz Euler and Unai Alegre-Ibarra; 
DNPU based network of devices to solve complex tasks 25/10/2019
'''

import torch
import numpy as np
import torch.nn as nn
from bspyproc.utils.pytorch import TorchUtils
from bspyproc.processors.processor_mgr import get_processor


class DNPUArchitecture(nn.Module):
    def __init__(self, configs):
        # offset min = -0.35 max = 0.7
        # scale min = 0.1 max = 1.5
        # conversion offset = -0.6
        super().__init__()
        self.conversion_offset = torch.tensor(configs['offset']['conversion'])
        self.offset = self.init_offset(configs['offset']['min'], configs['offset']['max'])
        self.scale = self.init_scale(configs['scale']['min'], configs['scale']['max'])
        self.configs = configs

    def init_offset(self, offset_min, offset_max):
        offset = offset_min + offset_max * np.random.rand(1, 2)
        offset = TorchUtils.get_tensor_from_numpy(offset)
        return nn.Parameter(offset)

    def init_scale(self, scale_min, scale_max):
        scale = scale_min + scale_max * np.random.rand(1)
        scale = TorchUtils.get_tensor_from_numpy(scale)
        return nn.Parameter(scale)

    def offset_penalty(self):
        return torch.sum(torch.relu(self.configs['offset']['min'] - self.offset) + torch.relu(self.offset - self.configs['offset']['max']))

    def scale_penalty(self):
        return torch.sum(torch.relu(self.configs['scale']['min'] - self.scale) + torch.relu(self.scale - self.configs['scale']['max']))


class TwoToOneDNPU(DNPUArchitecture):

    def __init__(self, configs):  # in_dict, path=r'../Data/Models/checkpoint3000_02-07-23h47m.pt'
        super().__init__(configs)
        self.init_model(configs)

    def init_model(self, configs):
        self.input_node1 = get_processor(configs)  # DNPU(in_dict['input_node1'], path=path)

        self.input_node2 = get_processor(configs)
        self.bn1 = nn.BatchNorm1d(2, affine=False)
        self.output_node = get_processor(configs)

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


class TwoToTwoToOneDNPU(DNPUArchitecture):
    def __init__(self, configs):
        super().__init__(configs)
        self.init_model(configs)

    def init_model(self, configs):
        self.input_node1 = get_processor(configs)  # DNPU(in_dict['input_node1'], path=path)
        self.input_node2 = get_processor(configs)  # DNPU(in_dict['input_node2'], path=path)
        self.bn1 = nn.BatchNorm1d(2, affine=False)

        self.hidden_node1 = get_processor(configs)  # DNPU(in_dict['hidden_node1'], path=path)
        self.hidden_node2 = get_processor(configs)  # DNPU(in_dict['hidden_node2'], path=path)
        self.bn2 = nn.BatchNorm1d(2, affine=False)

        self.output_node = get_processor(configs)  # DNPU(in_dict['output_node'], path=path)

    def forward(self, x):
        # Pass through input layer
        x = x + self.offset
        x1 = self.input_node1(x)
        x2 = self.input_node2(x)
        # --- BatchNorm --- #
        h = self.bn1(torch.cat((x1, x2), dim=1))
        std1 = np.sqrt(torch.mean(self.bn1.running_var).cpu().numpy())
        cut = 2 * std1
        # Pass through first hidden layer
        h = torch.tensor(1.8 / (4 * std1)) * \
            torch.clamp(h, min=-cut, max=cut) + self.conversion_offset
        h1 = self.hidden_node1(h)
        h2 = self.hidden_node2(h)
        # --- BatchNorm --- #
        h = self.bn2(torch.cat((h1, h2), dim=1))
        std2 = np.sqrt(torch.mean(self.bn2.running_var).cpu().numpy())
        cut = 2 * std2
        # Pass it through output layer
        h = torch.tensor(1.8 / (4 * std2)) * \
            torch.clamp(h, min=-cut, max=cut) + self.conversion_offset
        return self.output_node(h)

    def regularizer(self):
        control_penalty = self.input_node1.regularizer() \
            + self.input_node2.regularizer() \
            + self.output_node.regularizer()
        return control_penalty + self.offset_penalty() + self.scale_penalty()
