import torch
import numpy as np
from bspyproc.utils.waveform import generate_waveform
from bspyproc.utils.pytorch import TorchUtils


def get_control_voltage_indices(input_indices, length):
    return np.delete(np.arange(length), input_indices)


def merge_inputs_and_control_voltages(result, inputs, control_voltages, input_indices, control_voltage_indices):
    result[:, input_indices] = inputs
    result[:, control_voltage_indices] = control_voltages
    return result


def merge_inputs_and_control_voltages_in_numpy(inputs, control_voltages, input_indices, control_voltage_indices):
    result = np.empty((inputs.shape[0], len(input_indices) + len(control_voltage_indices)))
    return merge_inputs_and_control_voltages(result, inputs, control_voltages, input_indices, control_voltage_indices)


def merge_inputs_and_control_voltages_in_torch(inputs, control_voltages, input_indices, control_voltage_indices):
    result = TorchUtils.format_tensor(torch.empty((inputs.shape[0], len(input_indices) + len(control_voltage_indices))))
    return merge_inputs_and_control_voltages(result, inputs, control_voltages, input_indices, control_voltage_indices)
