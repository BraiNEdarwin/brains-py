import numpy as np

from brainspy.utils.pytorch import TorchUtils


def merge_electrode_data(
    inputs, control_voltages, input_indices, control_voltage_indices, use_torch=True
):
    result = np.empty(
        (inputs.shape[0], len(input_indices) + len(control_voltage_indices))
    )
    if use_torch:
        result = TorchUtils.get_tensor_from_numpy(result)
    result[:, input_indices] = inputs
    result[:, control_voltage_indices] = control_voltages
    return result


def transform_to_voltage(x_val,v_min,v_max,x_min,x_max):
    w,b = get_map_to_voltage_vars(v_min, v_max, x_min, x_max)
    return (x_val * w) + b

def get_map_to_voltage_vars(v_min, v_max, x_min, x_max):
    return get_scale(v_min,v_max,x_min,x_max), get_offset(v_min,v_max,x_min,x_max)

def get_scale(v_min, v_max, x_min, x_max):
    v = v_min - v_max
    x = x_min - x_max
    return v / x

def get_offset(v_min, v_max, x_min, x_max):
    v = (v_max * x_min) - (v_min * x_max)
    x = x_min - x_max
    return v / x