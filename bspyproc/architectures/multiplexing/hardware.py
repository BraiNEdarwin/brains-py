
# import torch
# import torch.nn as nn
import numpy as np
import os
from bspyproc.processors.processor_mgr import get_processor
from bspyproc.utils.waveform import generate_waveform, generate_slopped_plato
# from bspyproc.utils.pytorch import TorchUtils
from bspyproc.utils.control import get_control_voltage_indices
from bspyproc.utils.pytorch import TorchUtils
from bspyproc.utils.waveform import generate_slopped_plato, generate_waveform, generate_waveform_from_masked_data


class ArchitectureProcessor():
    def __init__(self, configs):
        self.configs = configs
        self.processor = get_processor(configs)
        self.clipping_value = configs['waveform']['output_clipping_value'] * self.get_amplification_value()
        # self.conversion_offset = configs['current_to_voltage']['offset']
        self.control_voltage_indices = get_control_voltage_indices(configs['input_indices'], configs['input_electrode_no'])

        if configs['batch_norm']['use_running_stats']:
            self.batch_norm_operation = self.batch_norm_running_stats
        else:
            self.batch_norm_operation = self.batch_norm_batch_stats

    def init_dirs(self, base_dir, is_main=True):
        if self.configs['debug']:
            if is_main:
                self.output_path = os.path.join(base_dir, 'validation', 'task_debug', 'hardware')
            else:
                self.output_path = os.path.join(base_dir, 'debug', 'hardware')
            if not os.path.exists(self.output_path):
                os.makedirs(self.output_path)

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

    def debug_batch_norm(self, x, mean, var):
        print(f"Using dataset mean and var: {self.configs['batch_norm']['use_running_stats']}")
        print(f'Running mean: {mean}')
        print(f'Current batch mean: {np.mean(x)}')
        print(f'Running var: {var}')
        print(f'Current batch mean: {np.var(x)}')

    def batch_norm(self, x, mean, var):
        self.debug_batch_norm(x, mean, var)
        return self.batch_norm_operation(x, mean, var)
        # return (x - mean) / np.sqrt(var + 1e-05)

    def batch_norm_batch_stats(self, x, mean, var):
        return (x - np.mean(x)) / np.sqrt(np.var(x) + 1e-05)

    def batch_norm_running_stats(self, x, mean, var):
        return (x - mean) / np.sqrt(var + 1e-05)

    # def current_to_voltage(self, x, std):
    #     return (self.current_to_voltage_conversion_amplitude * x) + self.current_to_voltage_conversion_offset
        # cut = 2 * std
        # return (1.8 / (4 * std)) * self.clip(x, cut_min=-cut, cut_max=cut) + self.conversion_offset

    def get_amplification_value(self):
        return self.processor.get_amplification_value()

    def get_control_voltages(self, x):
        return x[self.control_voltage_indices]


class TwoToOneProcessor(ArchitectureProcessor):
    pass


class TwoToTwoToOneProcessor(ArchitectureProcessor):
    def __init__(self, configs):
        super().__init__(configs)
        self.input_indices = self.get_input_indices(configs['input_indices'])
        self.control_voltage_indices = get_control_voltage_indices(self.input_indices, configs['input_electrode_no'] * 5)

    # def process_input(self, x, input_no=2):
    #     inputs = (self.scale * x) + self.offset
    #     clipped_inputs = np.empty((inputs.shape))
    #     for i in range(input_no):
    #         clipped_inputs[:, i] = self.clip(inputs[:, i].copy(), cut_min=self.min_voltage[self.input_indices[i]], cut_max=self.max_voltage[self.input_indices[i]])
    #         if not (inputs[:, i] == clipped_inputs[:, i]).all():
    #             print(f'Warning. Scale {self.scale} and offset {self.offset} caused inputs to go off limits. Input has been clipped for the security of the device. ')
    #             inputs[:, i] = clipped_inputs[:, i]

    #     return generate_waveform(inputs, self.configs['waveform']['amplitude_lengths'], self.configs['waveform']['slope_lengths'])

    def process_control_voltages(self, shape):
        result = np.zeros([shape,len(self.control_voltages)])
        slopped_plato = generate_slopped_plato(self.configs['waveform']['slope_lengths'], shape)
        for i in range(len(self.control_voltages)):
            result[:,i] = self.control_voltages[i] * slopped_plato
        return result

    def merge_inputs_and_control_voltages(self, inputs, control_voltages, node_no=5, node_electrode_no=7):
        result = np.zeros((inputs.shape[0], len(self.input_indices * node_no) + len(self.control_voltage_indices)))
        result[:, self.input_indices[:2]] = inputs
        result[:, self.input_indices[2:4]] = inputs
        result[:, node_electrode_no + self.input_indices[0]] = inputs[:, 0]
        result[:, node_electrode_no + self.input_indices[1]] = inputs[:, 1]
        result[:, self.control_voltage_indices] = control_voltages

        return result

    def current_to_voltage(self, x, electrode):
        return (self.current_to_voltage_conversion_amplitude[electrode] * self.clip(x, cut_min=-self.cut, cut_max=self.cut)) + self.current_to_voltage_conversion_offset[electrode]

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

    def read_from_processor(self, x, layer, device):
        print('------------------------------------------')
        print(f'Processing layer {layer}, device {device}:')

        i = int(x.shape[0] / 2)  # Get a value from the middle, where the control voltage should be in its plato
        max_threshold = x[i][x[i] > self.max_voltage]
        min_threshold = x[i][x[i] < self.min_voltage]
        fine = True
        if max_threshold.size > 0:
            print(f'Maximum threshold traspassed')
            print(f'Value: {max_threshold}')
            print(f'Threshold: {self.max_voltage[x[i] > self.max_voltage]}')
        if min_threshold.size > 0:
            print(f'Minimum threshold traspassed')
            print(f'Value: {min_threshold}')
            print(f'Threshold: {self.min_voltage[x[i] < self.min_voltage]}')
        if min_threshold.size > 0 or max_threshold.size > 0:
            print('Control voltages: ')
            print(x[i])
            fine = False
        results = self.processor.get_output(x)
        if fine:
            print('Finished fine')
        print('------------------------------------------')
        return results

    def get_output(self, x):
        if self.configs['debug']:
            np.save(os.path.join(self.output_path, 'raw_input'), x)
        x1 = self.read_from_processor(x[:, 0:7], 1, 0)

        bnx1 = self.process_layer(x1, self.bn1, 1, 0, self.configs['input_indices'][0])
        x[:, 14 + self.configs['input_indices'][0]] = bnx1
        x[:, 21 + self.configs['input_indices'][0]] = bnx1

        x2 = self.read_from_processor(x[:, 7:14], 1, 1)
        bnx2 = self.process_layer(x2, self.bn1, 1, 1, self.configs['input_indices'][1])
        # bnx1, bnx2 = self.process_layer(x1, x2, self.bn1, 1)

        x[:, 14 + self.configs['input_indices'][1]] = bnx2
        x[:, 21 + self.configs['input_indices'][1]] = bnx2

        h1 = self.read_from_processor(x[:, 14:21], 2, 0)
        bnx1 = self.process_layer(h1, self.bn2, 2, 0, self.configs['input_indices'][0])
        x[:, 28 + self.configs['input_indices'][0]] = bnx1
        # bnx1, bnx2 = self.process_layer(h1, h2, self.bn2, 2)

        h2 = self.read_from_processor(x[:, 21:28], 2, 1)
        bnx2 = self.process_layer(h2, self.bn2, 2, 1, self.configs['input_indices'][1])
        x[:, 28 + self.configs['input_indices'][1]] = bnx2

        return self.read_from_processor(x[:, 28:], 3, 0)

    def get_output_(self, inputs, mask):
        self.mask = mask
        # self.plato_indices = np.arange(len(mask))[mask]
        # inputs = self.process_inputs(inputs)
        x = self.merge_inputs_and_control_voltages(inputs, self.process_control_voltages(inputs.shape[0]))
        # x = merge_inputs_and_control_voltages_in_architecture(inputs, control_voltages, self.configs['input_indices'], self.control_voltage_indices, node_no=5, node_electrode_no=7, scale=self.scale, offset=self.offset, amplitudes=self.configs['waveform']['amplitude_lengths'], slopes=self.configs['waveform']['slope_lengths'])
        return self.get_output(x)

    def process_layer(self, x, bn, layer, device, electrode):
        # The input has been already scaled and offsetted
        if self.configs['debug']:
            np.save(os.path.join(self.output_path, 'device_layer_' + str(layer) + '_output_' + str(device)), x[:, 0])

        # Clip current
        x = self.clip(x, cut_min=-self.clipping_value, cut_max=self.clipping_value)
        if self.configs['debug']:
            np.save(os.path.join(self.output_path, 'bn_afterclip_' + str(layer) + '_' + str(device)), x[:, 0])

        # Batch normalisation
        bnx = self.batch_norm(x[self.mask], bn['mean'][device], bn['var'][device])
        if self.configs['debug']:
            np.save(os.path.join(self.output_path, 'bn_afterbatch_' + str(layer) + '_' + str(device)), generate_waveform_from_masked_data(bnx[:, 0], self.configs['waveform']['amplitude_lengths'], self.configs['waveform']['slope_lengths']))

        bnx = self.current_to_voltage(bnx, electrode)

        bnx = generate_waveform_from_masked_data(bnx[:, 0], self.configs['waveform']['amplitude_lengths'], self.configs['waveform']['slope_lengths'])
        if self.configs['debug']:
            np.save(os.path.join(self.output_path, 'bn_aftercv_' + str(layer) + '_' + str(device)), bnx)

        return bnx

    def set_batch_normalistaion_values(self, state_dict):
        self.bn1 = {}
        self.bn2 = {}
        self.bn1['mean'] = TorchUtils.get_numpy_from_tensor(state_dict['bn1.running_mean'])
        self.bn1['var'] = TorchUtils.get_numpy_from_tensor(state_dict['bn1.running_var'])
        self.bn1['batch_no'] = TorchUtils.get_numpy_from_tensor(state_dict['bn1.num_batches_tracked'])
        self.bn2['mean'] = TorchUtils.get_numpy_from_tensor(state_dict['bn2.running_mean'])
        self.bn2['var'] = TorchUtils.get_numpy_from_tensor(state_dict['bn2.running_var'])
        self.bn2['batch_no'] = TorchUtils.get_numpy_from_tensor(state_dict['bn2.num_batches_tracked'])

    # def set_scale_and_offset(self, state_dict):
    #     if 'scale' in state_dict:
    #         self.scale = TorchUtils.get_numpy_from_tensor(state_dict['scale'])
    #     else:
    #         self.scale = 1
    #     self.offset = TorchUtils.get_numpy_from_tensor(state_dict['offset'])

    def load_state_dict(self, model_dict):
        # self.set_scale_and_offset(model_dict['state_dict'])
        self.set_batch_normalistaion_values(model_dict)
        self.set_control_voltages(model_dict)
        self.set_current_to_voltage_conversion_params(model_dict['info'])
        self.model_dict = model_dict

    def set_current_to_voltage_conversion_params(self, state_dict):
        electrode_offset = np.asarray(state_dict['data_info']['input_data']['offset'])
        electrode_amplitude = np.asarray(state_dict['data_info']['input_data']['amplitude'])
        self.min_voltage = electrode_offset - electrode_amplitude
        self.max_voltage = electrode_offset + electrode_amplitude
        self.std = 1
        self.cut = 2 * self.std
        self.current_to_voltage_conversion_amplitude = (self.min_voltage - self.max_voltage) / (-4 * self.std)
        self.current_to_voltage_conversion_offset = ((((2 * self.std) - 1) / (2 * self.std)) * self.max_voltage) + (self.min_voltage / (2 * self.std))
        self.amplification = np.asarray(state_dict['data_info']['processor']['amplification'])

    def set_control_voltages(self, state_dict):
        control_voltages = np.zeros([len(self.control_voltage_indices)])
        control_voltages[0:5] = TorchUtils.get_numpy_from_tensor(state_dict['input_node1.bias'])
        control_voltages[5:10] = TorchUtils.get_numpy_from_tensor(state_dict['input_node2.bias'])
        control_voltages[10:15] = TorchUtils.get_numpy_from_tensor(state_dict['hidden_node1.bias'])
        control_voltages[15:20] = TorchUtils.get_numpy_from_tensor(state_dict['hidden_node2.bias'])
        control_voltages[20:25] = TorchUtils.get_numpy_from_tensor(state_dict['output_node.bias'])
        self.control_voltages = control_voltages
