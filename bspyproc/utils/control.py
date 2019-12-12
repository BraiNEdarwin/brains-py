import numpy as np


def get_control_voltage_indices(input_indices, length):
    return np.delete(np.arange(length), input_indices)


def merge_inputs_and_control_voltages(inputs, control_voltages, input_indices, control_voltage_indices):
    result = np.empty((inputs.shape[0], len(input_indices) + len(control_voltage_indices)))
    result[:, input_indices] = inputs
    result[:, control_voltage_indices] = control_voltages

    return result
