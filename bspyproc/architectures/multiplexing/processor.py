
# import torch
# import torch.nn as nn
import numpy as np
import os
from bspyproc.processors.processor_mgr import get_processor
from bspyproc.utils.waveform import generate_waveform
# from bspyproc.utils.pytorch import TorchUtils
from bspyproc.utils.control import get_control_voltage_indices, merge_inputs_and_control_voltages_in_architecture
from bspyproc.utils.pytorch import TorchUtils
from bspyproc.utils.waveform import generate_slopped_plato, generate_waveform, generate_waveform_from_masked_data


class ArchitectureProcessor():
    def __init__(self, configs):
        self.configs = configs
        self.processor = get_processor(configs)
        self.clipping_value = configs['waveform']['output_clipping_value'] * self.get_amplification_value()
        self.conversion_offset = configs['current_to_voltage']['offset']
        self.control_voltage_indices = get_control_voltage_indices(configs['input_indices'], configs['input_electrode_no'])
        self.output_path = 'tmp'

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
        # return (x - np.mean(x)) / np.sqrt(np.var(x) + 1e-05)
        return (x - mean) / np.sqrt(var + 1e-05)

    def current_to_voltage(self, x, std):
        cut = 2 * std
        return (1.8 / (4 * std)) * self.clip(x, cut_min=-cut, cut_max=cut) + self.conversion_offset

    def get_amplification_value(self):
        return self.processor.get_amplification_value()

    def get_control_voltages(self, x):
        return x[self.control_voltage_indices]


class TwoToOneProcessor(ArchitectureProcessor):
    pass
    # def __init__(self, configs):
    #     super().__init__(configs)
    #     self.input_indices = self.get_input_indices(configs['input_indices'])
    #     self.control_voltage_indices = get_control_voltage_indices(self.input_indices, configs['input_electrode_no'])
    #     if configs['batch_norm']:
    #         # self.bn1=nn.BatchNorm1d(2, affine=False)
    #         self.process_layer1 = self.process_layer1_batch_norm
    #     else:
    #         self.process_layer1 = self.process_layer1_alone

    # def get_input_indices(self, input_indices):
    #     result = np.empty(len(input_indices) * 3)
    #     result[0:7] = input_indices
    #     result[7:14] = input_indices + 7
    #     result[14:] = input_indices + 14
    #     return result

    # def get_output(self, x):
    #     x1 = self.processor.get_output(x[:, 0:7])
    #     x2 = self.processor.get_output(x[:, 7:14])
    #     x = self.process_layer1(x, x1, x2)

    #     self.control_voltages = self.get_control_voltages(x)
    #     x = self.processor.get_output(x[:, 14:])
    #     return self.process_output_layer(x)

    # def get_output_(self, inputs, control_voltages):
    #     slopped_plato = generate_slopped_plato(
    #         self.configs['waveform']['slope_lengths'], inputs.shape[0])[np.newaxis, :]
    #     control_voltages = slopped_plato * control_voltages[:, np.newaxis]

    #     x = merge_inputs_and_control_voltages_in_architecture(inputs, control_voltages, self.configs['input_indices'], self.control_voltage_indices, node_no=3, node_electrode_no=7, offset=self.offset, scale=self.scale,amplitudes=self.configs['waveform']['amplitude_lengths'], slopes=self.configs['waveform']['slope_lengths'])
    #     return self.get_output(x)

    # def process_layer1_alone(self, x, x1, x2):
    #     x[:, 14 + self.input_indices[0]] = self.clip(x1[:, 0], cut_min=-self.clipping_value, cut_max=self.clipping_value)
    #     x[:, 14 + self.input_indices[1]] = self.clip(x2[:, 0], cut_min=-self.clipping_value, cut_max=self.clipping_value)
    #     return x

    # def process_layer1_batch_norm(self, x, x1, x2):
    #     bnx = self.batch_norm(self.bn1, x1, x2)

    #     x[:, 14 + self.input_indices[0]] = self.clip(bnx[:, 0])
    #     x[:, 14 + self.input_indices[1]] = self.clip(bnx[:, 1])
    #     return x

    # def process_output_layer(self, y):
    #     return self.clip(y)

    # def set_batch_normalistaion_values(self, bn_statistics):
    #     self.bn1 = bn_statistics['bn_1']


class TwoToTwoToOneProcessor(ArchitectureProcessor):
    def __init__(self, configs):
        super().__init__(configs)
        self.input_indices = self.get_input_indices(configs['input_indices'])
        self.control_voltage_indices = get_control_voltage_indices(self.input_indices, configs['input_electrode_no'] * 5)

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
        np.save(os.path.join(self.output_path, 'raw_input'), x)
        x1 = self.processor.get_output(x[:, 0:7])
        x2 = self.processor.get_output(x[:, 7:14])

        bnx1, bnx2 = self.process_layer(x1, x2, self.bn1, 1)

        x[:, 14 + self.configs['input_indices'][0]] = bnx1
        x[:, 14 + self.configs['input_indices'][1]] = bnx2
        x[:, 21 + self.configs['input_indices'][0]] = bnx1
        x[:, 21 + self.configs['input_indices'][1]] = bnx2

        h1 = self.processor.get_output(x[:, 14:21])
        h2 = self.processor.get_output(x[:, 21:28])

        bnx1, bnx2 = self.process_layer(h1, h2, self.bn2, 2)
        x[:, 28 + self.configs['input_indices'][0]] = bnx1
        x[:, 28 + self.configs['input_indices'][1]] = bnx2

        return self.processor.get_output(x[:, 28:])

    def get_output_(self, inputs, mask):
        self.mask = mask
        self.plato_indices = np.arange(len(mask))[mask]
        control_voltages = np.linspace(self.control_voltages, self.control_voltages, inputs.shape[0])

        x = merge_inputs_and_control_voltages_in_architecture(inputs, control_voltages, self.configs['input_indices'], self.control_voltage_indices, node_no=5, node_electrode_no=7, scale=self.scale, offset=self.offset, amplitudes=self.configs['waveform']['amplitude_lengths'], slopes=self.configs['waveform']['slope_lengths'])
        return self.get_output(x)

    def process_layer(self, x1, x2, bn, layer):
        # The input has been already scaled and offsetted
        np.save(os.path.join(self.output_path, 'device_layer_' + str(layer) + '_output_1'), x1[:, 0])
        np.save(os.path.join(self.output_path, 'device_layer_' + str(layer) + '_output_2'), x2[:, 0])

        # Clip current
        x1 = self.clip(x1, cut_min=-self.clipping_value, cut_max=self.clipping_value)
        x2 = self.clip(x2, cut_min=-self.clipping_value, cut_max=self.clipping_value)
        np.save(os.path.join(self.output_path, 'bn_afterclip_' + str(layer) + '_1'), x1[:, 0])
        np.save(os.path.join(self.output_path, 'bn_afterclip_' + str(layer) + '_2'), x2[:, 0])

        # Batch normalisation
        bnx1 = self.batch_norm(x1[self.mask], bn['mean'][0], bn['var'][0])
        bnx2 = self.batch_norm(x2[self.mask], bn['mean'][1], bn['var'][1])

        # Transform current to voltage
        bnx1 = self.current_to_voltage(bnx1, np.sqrt(bn['var'][0]))
        bnx2 = self.current_to_voltage(bnx2, np.sqrt(bn['var'][1]))

        bnx1 = generate_waveform_from_masked_data(bnx1[:, 0], self.configs['waveform']['amplitude_lengths'], self.configs['waveform']['slope_lengths'])
        bnx2 = generate_waveform_from_masked_data(bnx2[:, 0], self.configs['waveform']['amplitude_lengths'], self.configs['waveform']['slope_lengths'])

        np.save(os.path.join(self.output_path, 'bn_aftercv_' + str(layer) + '_1'), bnx1)
        np.save(os.path.join(self.output_path, 'bn_aftercv_' + str(layer) + '_2'), bnx2)

        return bnx1, bnx2

    def set_batch_normalistaion_values(self, state_dict):
        self.bn1 = {}
        self.bn2 = {}
        self.bn1['mean'] = TorchUtils.get_numpy_from_tensor(state_dict['bn1.running_mean'])
        self.bn1['var'] = TorchUtils.get_numpy_from_tensor(state_dict['bn1.running_var'])
        self.bn1['batch_no'] = TorchUtils.get_numpy_from_tensor(state_dict['bn1.num_batches_tracked'])
        self.bn2['mean'] = TorchUtils.get_numpy_from_tensor(state_dict['bn2.running_mean'])
        self.bn2['var'] = TorchUtils.get_numpy_from_tensor(state_dict['bn2.running_var'])
        self.bn2['batch_no'] = TorchUtils.get_numpy_from_tensor(state_dict['bn2.num_batches_tracked'])

        # self.bn_statistics = bn_statistics
        # self.bn1 = bn_statistics['bn_1']
        # self.bn2 = bn_statistics['bn_2']

    def set_scale_and_offset(self, state_dict):
        if 'scale' in state_dict:
            self.scale = TorchUtils.get_numpy_from_tensor(state_dict['scale'])
        else:
            self.scale = 1
        self.offset = TorchUtils.get_numpy_from_tensor(state_dict['offset'])

    def load_state_dict(self, state_dict):
        self.set_scale_and_offset(state_dict)
        self.set_batch_normalistaion_values(state_dict)
        self.set_control_voltages(state_dict)
        self.state_dict = state_dict

    def set_control_voltages(self, state_dict):
        control_voltages = np.zeros([len(self.control_voltage_indices)])
        control_voltages[0:5] = TorchUtils.get_numpy_from_tensor(state_dict['input_node1.bias'])
        control_voltages[5:10] = TorchUtils.get_numpy_from_tensor(state_dict['input_node2.bias'])
        control_voltages[10:15] = TorchUtils.get_numpy_from_tensor(state_dict['hidden_node1.bias'])
        control_voltages[15:20] = TorchUtils.get_numpy_from_tensor(state_dict['hidden_node2.bias'])
        control_voltages[20:25] = TorchUtils.get_numpy_from_tensor(state_dict['output_node.bias'])
        self.control_voltages = control_voltages
