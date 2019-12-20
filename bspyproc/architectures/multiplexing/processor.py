
# import torch
# import torch.nn as nn
import numpy as np
from bspyproc.processors.processor_mgr import get_processor
from bspyproc.utils.waveform import generate_waveform
# from bspyproc.utils.pytorch import TorchUtils
from bspyproc.utils.control import get_control_voltage_indices, merge_inputs_and_control_voltages_in_architecture


class ArchitectureProcessor():
    def __init__(self, configs):
        self.configs = configs
        self.processor = get_processor(configs)
        self.clipping_value = configs['waveform']['output_clipping_value'] * self.get_amplification_value()
        self.conversion_offset = -0.6
        self.control_voltage_indices = get_control_voltage_indices(configs['input_indices'], configs['input_electrode_no'])

    def clip(self, x, cut_min, cut_max):
        x[x > cut_max] = cut_max
        x[x < cut_min] = cut_min
        return x

    # def batch_norm(self, bn, x1, x2):
    #     x1 = TorchUtils.get_tensor_from_numpy(x1)
    #     x2 = TorchUtils.get_tensor_from_numpy(x2)
    #     h = bn(torch.cat((x1, x2), dim=1))
    #     std1 = np.sqrt(torch.mean(bn.running_var).cpu().numpy())
    #     cut = 2 * std1
    #     # Pass it through output layer
    #     h = torch.tensor(1.8 / (4 * std1)) * \
    #         torch.clamp(h, min=-cut, max=cut) + self.conversion_offset
    #     return h.numpy()

    def batch_norm(self, x, mean, var):
        return (x - mean) / np.sqrt(var)

    def current_to_voltage(self, x, var):
        std = np.sqrt(var)
        cut = 2 * std
        h = (1.8 / (4 * std)) * self.clip(x, cut_min=-cut, cut_max=cut)
        return h + self.conversion_offset

    def get_amplification_value(self):
        return self.processor.get_amplification_value()

    def get_control_voltages(self, x):
        return x[self.control_voltage_indices]


class TwoToOneProcessor(ArchitectureProcessor):
    def __init__(self, configs):
        super().__init__(configs)
        self.input_indices = self.get_input_indices(configs['input_indices'])
        self.control_voltage_indices = get_control_voltage_indices(self.input_indices, configs['input_electrode_no'])
        if configs['batch_norm']:
            # self.bn1=nn.BatchNorm1d(2, affine=False)
            self.process_layer1 = self.process_layer1_batch_norm
        else:
            self.process_layer1 = self.process_layer1_alone

    def get_input_indices(self, input_indices):
        result = np.empty(len(input_indices) * 3)
        result[0:7] = input_indices
        result[7:14] = input_indices + 7
        result[14:] = input_indices + 14
        return result

    def get_output(self, x):
        x1 = self.processor.get_output(x[:, 0:7])
        x2 = self.processor.get_output(x[:, 7:14])
        x = self.process_layer1(x, x1, x2)

        self.control_voltages = self.get_control_voltages(x)
        x = self.processor.get_output(x[:, 14:])
        return self.process_output_layer(x)

    def get_output_(self, inputs, control_voltages):
        x = merge_inputs_and_control_voltages_in_architecture(inputs, control_voltages, self.configs['input_indices'], self.control_voltage_indices, node_no=3, node_electrode_no=7)
        return self.get_output(x)

    def process_layer1_alone(self, x, x1, x2):
        x[:, 14 + self.input_indices[0]] = self.clip(x1[:, 0], cut_min=-self.clipping_value, cut_max=self.clipping_value)
        x[:, 14 + self.input_indices[1]] = self.clip(x2[:, 0], cut_min=-self.clipping_value, cut_max=self.clipping_value)
        return x

    def process_layer1_batch_norm(self, x, x1, x2):
        bnx = self.batch_norm(self.bn1, x1, x2)

        x[:, 14 + self.input_indices[0]] = self.clip(bnx[:, 0])
        x[:, 14 + self.input_indices[1]] = self.clip(bnx[:, 1])
        return x

    def process_output_layer(self, y):
        return self.clip(y)

    def set_batch_normalistaion_values(self, bn_statistics):
        self.bn1 = bn_statistics['bn_1']


class TwoToTwoToOneProcessor(ArchitectureProcessor):
    def __init__(self, configs):
        super().__init__(configs)
        self.input_indices = self.get_input_indices(configs['input_indices'])
        self.control_voltage_indices = get_control_voltage_indices(self.input_indices, configs['input_electrode_no'] * 5)
        if configs['batch_norm']:
            # self.bn1 = nn.BatchNorm1d(2, affine=False)
            # self.bn2 = nn.BatchNorm1d(2, affine=False)
            self.process_layer1 = self.process_layer1_batch_norm
            self.process_layer2 = self.process_layer2_batch_norm
        else:
            self.process_layer1 = self.process_layer1_alone
            self.process_layer2 = self.process_layer2_alone

    def get_input_indices(self, input_indices):
        result = np.zeros(len(input_indices) * 5, dtype=int)
        result[0] = input_indices[0]
        result[1] = input_indices[1]
        result[2] = input_indices[0] + 7
        result[3] = input_indices[1] + 7
        result[4] = input_indices[0] + 14
        result[5] = input_indices[1] + 14
        result[6] = input_indices[0] + 21
        result[7] = input_indices[1] + 21
        result[8] = input_indices[0] + 28
        result[9] = input_indices[1] + 28
        return result

    def get_output(self, x):

        x1 = self.processor.get_output(x[:, 0:7])
        x2 = self.processor.get_output(x[:, 7:14])
        x = self.process_layer1(x, x1, x2)

        h1 = self.processor.get_output(x[:, 14:21])
        h2 = self.processor.get_output(x[:, 21:28])
        x = self.process_layer2(x, h1, h2)

        self.control_voltages = self.get_control_voltages(x)

        return self.processor.get_output(x[:, 28:])

    def get_output_(self, inputs, control_voltages, mask):
        self.mask = mask
        self.plato_indices = np.arange(len(mask))[mask]
        x = merge_inputs_and_control_voltages_in_architecture(inputs, control_voltages, self.configs['input_indices'], self.control_voltage_indices, node_no=5, node_electrode_no=7, scale=self.scale, offset=self.offset)
        return self.get_output(x)

    def process_layer1_alone(self, x, x1, x2):
        # x[:, 14 + self.configs['input_indices'][0]] = self.current_to_voltage(self.clip(x1[:, 0], cut_min=-self.clipping_value, cut_max=self.clipping_value))
        # x[:, 14 + self.configs['input_indices'][1]] = self.current_to_voltage(self.clip(x2[:, 0], cut_min=-self.clipping_value, cut_max=self.clipping_value))
        # x[:, 21 + self.configs['input_indices'][0]] = self.current_to_voltage(self.clip(x1[:, 0], cut_min=-self.clipping_value, cut_max=self.clipping_value))
        # x[:, 21 + self.configs['input_indices'][1]] = self.current_to_voltage(self.clip(x2[:, 0], cut_min=-self.clipping_value, cut_max=self.clipping_value))
        return x

    def process_layer1_batch_norm(self, x, x1, x2):
        # The input has been already scaled and offsetted

        # Clip current
        x1 = self.clip(x1, cut_min=-self.clipping_value, cut_max=self.clipping_value)
        x2 = self.clip(x2, cut_min=-self.clipping_value, cut_max=self.clipping_value)

        # Batch normalisation
        bnx1 = self.batch_norm(x1[self.mask], self.bn1['mean'][0], self.bn1['var'][0])
        bnx2 = self.batch_norm(x2[self.mask], self.bn1['mean'][1], self.bn1['var'][1])

        # Get mean of platos and create waveform back
        bnx1 = self.process_batch_norm(bnx1[:, 0])
        bnx2 = self.process_batch_norm(bnx2[:, 0])

        # Convert from current to voltage, clip voltage, and save into corresponding indices
        x[:, 14 + self.configs['input_indices'][0]] = self.current_to_voltage(bnx1, self.bn1['var'][0])
        x[:, 14 + self.configs['input_indices'][1]] = self.current_to_voltage(bnx2, self.bn1['var'][1])
        x[:, 21 + self.configs['input_indices'][0]] = self.current_to_voltage(bnx1, self.bn1['var'][0])
        x[:, 21 + self.configs['input_indices'][1]] = self.current_to_voltage(bnx2, self.bn1['var'][1])

        return x

    def process_batch_norm(self, bnx):
        i = 0
        amplitudes = np.array([])
        while i < len(bnx):
            aux = bnx[i:i + self.configs['waveform']['amplitude_lengths']]
            amplitudes = np.append(amplitudes, np.mean(aux))
            i += self.configs['waveform']['amplitude_lengths']
        return generate_waveform(amplitudes, self.configs['waveform']['amplitude_lengths'], self.configs['waveform']['slope_lengths'])

    def process_layer2_alone(self, x, x1, x2):
        # x[:, 28 + self.configs['input_indices'][0]] = self.current_to_voltage(self.clip(x1[:, 0], cut_min=-self.clipping_value, cut_max=self.clipping_value))
        # x[:, 28 + self.configs['input_indices'][1]] = self.current_to_voltage(self.clip(x2[:, 0], cut_min=-self.clipping_value, cut_max=self.clipping_value))
        return x

    def process_layer2_batch_norm(self, x, x1, x2):
        # Clip current
        x1 = self.clip(x1, cut_min=-self.clipping_value, cut_max=self.clipping_value)
        x2 = self.clip(x2, cut_min=-self.clipping_value, cut_max=self.clipping_value)

        # Batch normalisation
        bnx1 = self.batch_norm(x1[self.mask], self.bn2['mean'][0], self.bn2['var'][0])
        bnx2 = self.batch_norm(x2[self.mask], self.bn2['mean'][1], self.bn2['var'][1])

        # Get mean of platos and create waveform back
        bnx1 = self.process_batch_norm(bnx1[:, 0])
        bnx2 = self.process_batch_norm(bnx2[:, 0])

        # Convert from current to voltage, clip voltage, and save into corresponding indices
        x[:, 28 + self.configs['input_indices'][0]] = self.current_to_voltage(bnx1, self.bn2['var'][0])
        x[:, 28 + self.configs['input_indices'][1]] = self.current_to_voltage(bnx2, self.bn2['var'][1])

        return x

    def set_batch_normalistaion_values(self, bn_statistics):
        self.bn1 = bn_statistics['bn_1']
        self.bn2 = bn_statistics['bn_2']

    def set_scale_and_offset(self, scale, offset):
        self.scale = scale
        self.offset = offset
