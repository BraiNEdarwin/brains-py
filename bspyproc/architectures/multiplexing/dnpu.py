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

    def batch_norm(self, bn, x1, x2):
        h = bn(torch.cat((x1, x2), dim=1))
        std1 = np.sqrt(torch.mean(bn.running_var).cpu().numpy())
        cut = 2 * std1
        # Pass it through output layer
        h = torch.tensor(1.8 / (4 * std1)) * \
            self.clip(h, cut) + self.conversion_offset
        return h

    def clip(self, x, clipping_value):
        return torch.clamp(x, min=-clipping_value, max=clipping_value)


class TwoToOneDNPU(DNPUArchitecture):

    def __init__(self, configs):
        super().__init__(configs)
        self.init_model(configs)
        self.init_clipping_values(configs['waveform']['output_clipping_value'])
        if configs['batch_norm']:
            self.bn1 = TorchUtils.format_tensor(nn.BatchNorm1d(2, affine=False))
            self.process_layer1 = self.process_layer1_batch_norm
        else:
            self.process_layer1 = self.process_layer1_alone

    def init_model(self, configs):
        self.input_node1 = get_processor(configs)  # DNPU(in_dict['input_node1'], path=path)
        self.input_node2 = get_processor(configs)
        self.output_node = get_processor(configs)

    def init_clipping_values(self, base_clipping_value):
        self.input_node1_clipping_value = base_clipping_value * self.input_node1.get_amplification_value()
        self.input_node2_clipping_value = base_clipping_value * self.input_node2.get_amplification_value()
        self.output_node_clipping_value = base_clipping_value * self.output_node.get_amplification_value()

    def forward(self, x):
        # Pass through input layer
        x = (self.scale * x) + self.offset

        x1 = self.input_node1(x)
        x2 = self.input_node2(x)
        x = self.process_layer1(x, x1, x2)

        x = self.output_node(x)
        return self.process_output_layer(x)

    def regularizer(self):
        control_penalty = self.input_node1.regularizer() \
            + self.input_node2.regularizer() \
            + self.output_node.regularizer()
        return control_penalty + self.offset_penalty() + self.scale_penalty()

    def process_layer1_alone(self, x, x1, x2):
        x[:, 0] = self.clip(x1[:, 0], self.input_node1_clipping_value)
        x[:, 1] = self.clip(x2[:, 0], self.input_node2_clipping_value)
        return x

    def process_layer1_batch_norm(self, x, x1, x2):
        bnx = self.batch_norm(self.bn1, x1, x2)

        x[:, 0] = self.clip(bnx[:, 0], self.input_node1_clipping_value)
        x[:, 1] = self.clip(bnx[:, 1], self.input_node2_clipping_value)

        return x

    def process_output_layer(self, y):
        return self.clip(y, self.output_node_clipping_value)

    def get_control_voltages(self):
        w1 = next(self.input_node1.parameters()).detach().cpu().numpy()
        w2 = next(self.input_node2.parameters()).detach().cpu().numpy()
        w3 = next(self.output_node.parameters()).detach().cpu().numpy()
        return torch.stack([w1, w2, w3])

    def get_bn_statistics(self):
        bn_statistics = {'bn_1': {}}
        bn_statistics['bn_1']['mean'] = self.bn1.running_mean.cpu().detach().numpy()
        bn_statistics['bn_1']['var'] = self.bn1.running_var.cpu().detach().numpy()
        return bn_statistics


class TwoToTwoToOneDNPU(DNPUArchitecture):
    def __init__(self, configs):
        super().__init__(configs)
        self.init_model(configs)
        self.init_clipping_values(configs['waveform']['output_clipping_value'])
        if configs['batch_norm']:
            self.bn1 = TorchUtils.format_tensor(nn.BatchNorm1d(2, affine=False))
            self.bn2 = TorchUtils.format_tensor(nn.BatchNorm1d(2, affine=False))
            self.process_layer1 = self.process_layer1_batch_norm
            self.process_layer2 = self.process_layer2_batch_norm
        else:
            self.process_layer1 = self.process_layer1_alone
            self.process_layer2 = self.process_layer2_alone

    def init_model(self, configs):
        self.input_node1 = get_processor(configs)  # DNPU(in_dict['input_node1'], path=path)
        self.input_node2 = get_processor(configs)  # DNPU(in_dict['input_node2'], path=path)

        self.hidden_node1 = get_processor(configs)  # DNPU(in_dict['hidden_node1'], path=path)
        self.hidden_node2 = get_processor(configs)  # DNPU(in_dict['hidden_node2'], path=path)

        self.output_node = get_processor(configs)  # DNPU(in_dict['output_node'], path=path)

    def init_clipping_values(self, base_clipping_value):
        self.input_node1_clipping_value = base_clipping_value * self.input_node1.get_amplification_value()
        self.input_node2_clipping_value = base_clipping_value * self.input_node2.get_amplification_value()
        self.hidden_node1_clipping_value = base_clipping_value * self.hidden_node1.get_amplification_value()
        self.hidden_node2_clipping_value = base_clipping_value * self.hidden_node2.get_amplification_value()
        self.output_node_clipping_value = base_clipping_value * self.output_node.get_amplification_value()

    def forward(self, x):
        # Pass through input layer
        x = (self.scale * x) + self.offset

        x1 = self.input_node1(x)
        x2 = self.input_node2(x)
        x = self.process_layer1(x, x1, x2)

        h1 = self.hidden_node1(x)
        h2 = self.hidden_node2(x)
        x = self.process_layer2(x, h1, h2)

        x = self.output_node(x)
        return self.process_output_layer(x)

    def regularizer(self):
        control_penalty = self.input_node1.regularizer() \
            + self.input_node2.regularizer() \
            + self.output_node.regularizer()
        return control_penalty + self.offset_penalty() + self.scale_penalty()

    def process_layer1_alone(self, x, x1, x2):
        x[:, 0] = self.clip(x1[:, 0], self.input_node1_clipping_value)
        x[:, 1] = self.clip(x2[:, 0], self.input_node2_clipping_value)
        return x

    def process_layer1_batch_norm(self, x, x1, x2):
        bnx = self.batch_norm(self.bn1, x1, x2)

        x[:, 0] = self.clip(bnx[:, 0], self.input_node1_clipping_value)
        x[:, 1] = self.clip(bnx[:, 1], self.input_node2_clipping_value)
        return x

    def process_layer2_alone(self, x, x1, x2):
        x[:, 0] = self.clip(x1[:, 0], self.hidden_node1_clipping_value)
        x[:, 1] = self.clip(x2[:, 0], self.hidden_node2_clipping_value)
        return x

    def process_layer2_batch_norm(self, x, x1, x2):
        bnx = self.batch_norm(self.bn2, x1, x2)

        x[:, 0] = self.clip(bnx[:, 0], self.hidden_node1_clipping_value)
        x[:, 1] = self.clip(bnx[:, 1], self.hidden_node2_clipping_value)
        return x

    def process_output_layer(self, y):
        return self.clip(y, self.output_node_clipping_value)

    def get_control_voltages(self):
        w1 = next(self.input_node1.parameters()).detach()[0, :]
        w2 = next(self.input_node2.parameters()).detach()[0, :]
        w3 = next(self.hidden_node1.parameters()).detach()[0, :]
        w4 = next(self.hidden_node2.parameters()).detach()[0, :]
        w5 = next(self.output_node.parameters()).detach()[0, :]
        return torch.stack([w1, w2, w3, w4, w5])

    def get_bn_statistics(self):
        bn_statistics = {'bn_1': {}, 'bn_2': {}}
        bn_statistics['bn_1']['mean'] = self.bn1.running_mean.cpu().detach().numpy()
        bn_statistics['bn_1']['var'] = self.bn1.running_var.cpu().detach().numpy()
        bn_statistics['bn_2']['mean'] = self.bn2.running_mean.cpu().detach().numpy()
        bn_statistics['bn_2']['var'] = self.bn2.running_var.cpu().detach().numpy()
        return bn_statistics


if __name__ == '__main__':

    from bspyalgo.utils.io import load_configs
    import matplotlib.pyplot as plt

    config_path = 'configs/configs_dnpu21.json'
    CONFIGS = load_configs(config_path)
    dnpu_221 = TwoToTwoToOneDNPU(CONFIGS)

    INPUTS = TorchUtils.get_tensor_from_numpy(np.random.rand(1000, 2))
    OUTPUS = dnpu_221(INPUTS)

    bn_statistics = dnpu_221.get_bn_statistics()
    print(bn_statistics)

    plt.hist(OUTPUS.cpu().detach().numpy())
    plt.show()
