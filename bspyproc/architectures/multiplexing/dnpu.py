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
        self.configs = configs
        self.conversion_offset = torch.tensor(configs['current_to_voltage']['offset'])
        self.offset = self.init_offset(configs['offset']['min'], configs['offset']['max'])
        self.scale = self.init_scale(configs['scale']['min'], configs['scale']['max'])

    def init_offset(self, offset_min, offset_max):
        offset = offset_min + offset_max * np.random.rand(1, 2)
        offset = TorchUtils.get_tensor_from_numpy(offset)
        return nn.Parameter(offset)

    def init_scale(self, scale_min, scale_max):
        if self.configs['scale']['min'] == 1.0 and self.configs['scale']['max'] == 1.0:
            scale = TorchUtils.get_tensor_from_numpy(np.array([1.0]))
            return scale
        else:
            scale = TorchUtils.get_tensor_from_numpy(scale_min + scale_max * np.random.rand(1))
            return nn.Parameter(scale)

    def offset_penalty(self):
        return torch.sum(torch.relu(self.configs['offset']['min'] - self.offset) + torch.relu(self.offset - self.configs['offset']['max']))

    def scale_penalty(self):
        return torch.sum(torch.relu(self.configs['scale']['min'] - self.scale) + torch.relu(self.scale - self.configs['scale']['max']))

    def batch_norm(self, bn, x1, x2):
        h = bn(torch.cat((x1, x2), dim=1))
        std = np.sqrt(bn.running_var.clone().cpu().numpy())
        return h, std

    def current_to_voltage(self, x, std):
        # Pass it through output layer and clip it to two times the standard deviation
        cut = 2 * std
        return torch.tensor(1.8 / (4 * std)) * self.clip(x, cut) + self.conversion_offset
        # torch.save(voltage,'voltage.pt')

    def clip(self, x, clipping_value):
        return torch.clamp(x, min=-clipping_value, max=clipping_value)


class TwoToOneDNPU(DNPUArchitecture):

    def __init__(self, configs):
        pass
#         super().__init__(configs)
#         self.init_model(configs)
#         self.init_clipping_values(configs['waveform']['output_clipping_value'])
#         if configs['batch_norm']:
#             self.bn1 = TorchUtils.format_tensor(nn.BatchNorm1d(2, affine=False))
#             self.process_layer1 = self.process_layer1_batch_norm
#         else:
#             self.process_layer1 = self.process_layer1_alone

#     def init_model(self, configs):
#         self.input_node1 = get_processor(configs)  # DNPU(in_dict['input_node1'], path=path)
#         self.input_node2 = get_processor(configs)
#         self.output_node = get_processor(configs)

#     def init_clipping_values(self, base_clipping_value):
#         self.input_node1_clipping_value = base_clipping_value * self.input_node1.get_amplification_value()
#         self.input_node2_clipping_value = base_clipping_value * self.input_node2.get_amplification_value()
#         self.output_node_clipping_value = base_clipping_value * self.output_node.get_amplification_value()

#     def forward(self, x):
#         # Pass through input layer
#         x = (self.scale * x) + self.offset

#         x1 = self.input_node1(x)
#         x2 = self.input_node2(x)
#         x = self.process_layer1(x, x1, x2)

#         x = self.output_node(x)
#         return self.process_output_layer(x)

#     def regularizer(self):
#         control_penalty = self.input_node1.regularizer() \
#             + self.input_node2.regularizer() \
#             + self.output_node.regularizer()
#         return control_penalty + self.offset_penalty() + self.scale_penalty()

#     def process_layer1_alone(self, x, x1, x2):
#         x[:, 0] = self.clip(x1[:, 0], self.input_node1_clipping_value)
#         x[:, 1] = self.clip(x2[:, 0], self.input_node2_clipping_value)
#         return x

#     def process_layer1_batch_norm(self, x, x1, x2):

#         bnx, std = self.batch_norm(self.bn1, x1, x2)
#         torch.save(bnx, f'bn_afterbatch_1.pt')

#         bnx = self.current_to_voltage(bnx, std)
#         torch.save(bnx, f'bn_aftercv_1.pt')

#         x[:, 0] = self.clip(bnx[:, 0], self.input_node1_clipping_value)
#         x[:, 1] = self.clip(bnx[:, 1], self.input_node2_clipping_value)

#         return x

#     def process_output_layer(self, y):
#         return self.clip(y, self.output_node_clipping_value)

#     def get_control_voltages(self):
#         w1 = next(self.input_node1.parameters()).detach().cpu().numpy()
#         w2 = next(self.input_node2.parameters()).detach().cpu().numpy()
#         w3 = next(self.output_node.parameters()).detach().cpu().numpy()
#         return torch.stack([w1, w2, w3])

#     def get_bn_statistics(self):
#         bn_statistics = {'bn_1': {}}
#         bn_statistics['bn_1']['mean'] = self.bn1.running_mean.cpu().detach().numpy()
#         bn_statistics['bn_1']['var'] = self.bn1.running_var.cpu().detach().numpy()
#         return bn_statistics


class TwoToTwoToOneDNPU(DNPUArchitecture):

    def __init__(self, configs):
        super().__init__(configs)
        self.init_model(configs)
        self.init_clipping_values(configs['waveform']['output_clipping_value'])
        self.bn1 = TorchUtils.format_tensor(nn.BatchNorm1d(2, affine=False))
        self.bn2 = TorchUtils.format_tensor(nn.BatchNorm1d(2, affine=False))

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

        # Scale and offset
        x = (self.scale * x) + self.offset
        torch.save(x, 'raw_input.pt')
        # Clipping and passing data to the first layer
        x = self.process_layer(self.input_node1(x), self.input_node2(x), self.bn1, self.input_node1_clipping_value, self.input_node2_clipping_value, 1)
        x = self.process_layer(self.hidden_node1(x), self.hidden_node2(x), self.bn2, self.hidden_node1_clipping_value, self.hidden_node2_clipping_value, 2)

        return self.output_node(x)

    def regularizer(self):
        control_penalty = self.input_node1.regularizer() \
            + self.input_node2.regularizer() \
            + self.output_node.regularizer()
        return control_penalty + self.offset_penalty() + self.scale_penalty()

    def process_layer(self, x1, x2, bn, clipping_value_1, clipping_value_2, i):
        torch.save(x1[:, 0], 'device_layer_' + str(i) + '_output_1.pt')
        torch.save(x2[:, 0], 'device_layer_' + str(i) + '_output_2.pt')
        # Clip values at 400
        x1 = self.clip(x1, clipping_value=clipping_value_1)
        x2 = self.clip(x2, clipping_value=clipping_value_2)
        torch.save(x1[:, 0], 'bn_afterclip_' + str(i) + '_1.pt')
        torch.save(x2[:, 0], 'bn_afterclip_' + str(i) + '_2.pt')

        bnx, std = self.batch_norm(bn, x1, x2)
        torch.save(bnx[:, 0], f'bn_afterbatch_' + str(i) + '_1.pt')
        torch.save(bnx[:, 1], f'bn_afterbatch_' + str(i) + '_2.pt')

        bnx_0 = self.current_to_voltage(bnx[:, 0], std[0])
        bnx_1 = self.current_to_voltage(bnx[:, 1], std[1])
        torch.save(bnx_0, f'bn_aftercv_' + str(i) + '_1.pt')
        torch.save(bnx_1, f'bn_aftercv_' + str(i) + '_2.pt')
        return torch.cat((bnx_0[:, None], bnx_1[:, None]), dim=1)

    def get_control_voltages(self):
        w1 = next(self.input_node1.parameters()).detach()[0, :]
        w2 = next(self.input_node2.parameters()).detach()[0, :]
        w3 = next(self.hidden_node1.parameters()).detach()[0, :]
        w4 = next(self.hidden_node2.parameters()).detach()[0, :]
        w5 = next(self.output_node.parameters()).detach()[0, :]
        return torch.stack([w1, w2, w3, w4, w5])

    def set_bn_dict(self, bn_dict):
        self.bn1.load_state_dict(bn_dict['bn_1'])
        self.bn2.load_state_dict(bn_dict['bn_2'])

    def get_bn_dict(self):
        bn_dict = {'bn_1': {}, 'bn_2': {}}
        bn_dict['bn_1'] = self.bn1.state_dict()
        bn_dict['bn_2'] = self.bn2.state_dict()
        return bn_dict

    def reset(self):
        # This function needs to be checked
        self.input_node1.reset()
        self.input_node2.reset()
        self.hidden_node1.reset()
        self.hidden_node2.reset()
        self.output_node.reset()
        self.offset.data.uniform_(self.configs['offset']['min'], self.configs['offset']['max'])
        self.scale = self.init_scale(self.configs['scale']['min'], self.configs['scale']['max'])
        if self.configs['batch_norm']:
            self.bn1 = TorchUtils.format_tensor(nn.BatchNorm1d(2, affine=False))
            self.bn2 = TorchUtils.format_tensor(nn.BatchNorm1d(2, affine=False))

    def get_scale(self):
        return TorchUtils.get_numpy_from_tensor(self.scale.detach())

    def get_offset(self):
        return TorchUtils.get_numpy_from_tensor(self.offset.detach())

    def initialise_parameters(self, control_voltages, bn_stats, scale, offset):
        self.set_control_voltages(control_voltages)
        self.set_scale_and_offset(scale, offset)
        self.set_bn_dict(bn_stats)

    def set_scale_and_offset(self, scale, offset):
        state_dict = self.state_dict()
        state_dict['scale'] = TorchUtils.get_tensor_from_numpy(scale)
        state_dict['offset'] = TorchUtils.get_tensor_from_numpy(offset)
        self.load_state_dict(state_dict)

    def set_control_voltages(self, control_voltages):
        state_dict = self.state_dict()
        state_dict['input_node1.bias'] = TorchUtils.get_tensor_from_numpy(control_voltages[np.newaxis, 0:5])
        state_dict['input_node2.bias'] = TorchUtils.get_tensor_from_numpy(control_voltages[np.newaxis, 5:10])
        state_dict['hidden_node1.bias'] = TorchUtils.get_tensor_from_numpy(control_voltages[np.newaxis, 10:15])
        state_dict['hidden_node2.bias'] = TorchUtils.get_tensor_from_numpy(control_voltages[np.newaxis, 15:20])
        state_dict['output_node.bias'] = TorchUtils.get_tensor_from_numpy(control_voltages[np.newaxis, 20:25])
        self.load_state_dict(state_dict)


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
