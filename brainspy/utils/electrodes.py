import numpy as np

from brainspy.utils.pytorch import TorchUtils


def merge_electrode_data(inputs, control_voltages, input_indices, control_voltage_indices, use_torch=True):
    result = np.empty((inputs.shape[0], len(input_indices) + len(control_voltage_indices)))
    if use_torch:
        result = TorchUtils.get_tensor_from_numpy(result)
    result[:, input_indices] = inputs
    result[:, control_voltage_indices] = control_voltages
    return result


def get_map_to_voltage_vars(v_min, v_max, x_min, x_max):
    scale = ((v_min - v_max) / (x_min - x_max))
    offset = v_max - scale * x_max
    return scale, offset
