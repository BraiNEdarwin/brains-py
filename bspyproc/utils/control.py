import numpy as np


def get_control_voltage_indices(input_indices, length):
    processor_input = np.arange(length)
    return np.setdiff1d(processor_input, np.array(input_indices))


def merge_inputs_and_control_voltages(inputs, control_voltages, input_indices, control_voltage_indices):
    result = np.empty((inputs.shape[0], len(input_indices) + len(control_voltage_indices)))
    result[:, input_indices] = inputs
    result[:, control_voltage_indices] = control_voltages

    return result
