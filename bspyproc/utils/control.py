import numpy as np


def get_control_voltage_indices(input_indices, length):
    return np.delete(np.arange(length), input_indices)


def merge_inputs_and_control_voltages(inputs, control_voltages, input_indices, control_voltage_indices):
    result = np.empty((inputs.shape[0], len(input_indices) + len(control_voltage_indices)))
    result[:, input_indices] = inputs
    result[:, control_voltage_indices] = control_voltages

    return result


def merge_inputs_and_control_voltages_in_architecture(inputs, control_voltages, input_indices, control_voltage_indices, node_no, node_electrode_no):
    result = np.empty((inputs.shape[0], len(input_indices * node_no) + len(control_voltage_indices)))
    result[:, input_indices] = inputs
    result[:, node_electrode_no + input_indices[0]] = inputs[:, 0]
    result[:, node_electrode_no + input_indices[1]] = inputs[:, 1]
    result[:, control_voltage_indices] = control_voltages

    return result
