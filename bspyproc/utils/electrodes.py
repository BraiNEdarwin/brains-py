import numpy as np

from bspyproc.utils.pytorch import TorchUtils


def merge_electrode_data(inputs, control_voltages, input_indices, control_voltage_indices, use_torch=True):
    result = np.empty((inputs.shape[0], len(input_indices) + len(control_voltage_indices)))
    if use_torch:
        result = TorchUtils.get_tensor_from_numpy(result)
    result[:, input_indices] = inputs
    result[:, control_voltage_indices] = control_voltages
    return result
