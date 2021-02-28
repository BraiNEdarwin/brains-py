import numpy as np

from brainspy.utils.pytorch import TorchUtils

# Used in processors/processor.py.
def merge_electrode_data(
    inputs, control_voltages, input_indices, control_voltage_indices, use_torch=True
):
    """
    Merge data from two electrodes with the specified indices for each.
    Need to indicate whether numpy or torch is used. The result will
    have the same type as the input.

    Example
    -------
    Let inputs = [i_1, i_2]  and control_voltages = [c_1, ..., c_5] where i_1, c_1, etc are column vectors.
    Let input_indices = [0, 2] and control_voltages = [3, 1, 4, 5, 6].
    Then this method would return [i_1, c_2, i_2, c_1, c_3, c_4, c_5].
    The result will be a numpy array or a torch tensor depending on the input.

    Parameters
    ----------
    inputs: np.array or torch.tensor
        Data for the input electrodes.
    control_voltages: np.array or torch.tensor
        Data for the control electrodes.
    input_indices: iterable of int
        Indices of the input electrodes.
    control_voltage_indices: iterable of int
        Indices of the control electrodes.
    use_torch : boolean
        Indicate whether the data is pytorch tensor (instead of a numpy array)

    Returns
    -------
    result: np.array or torch.tensor
        Array or tensor with merged data.

    """
    result = np.empty(
        (inputs.shape[0], len(input_indices) + len(control_voltage_indices))
    )
    if use_torch:
        result = TorchUtils.get_tensor_from_numpy(result)
    result[:, input_indices] = inputs
    result[:, control_voltage_indices] = control_voltages
    return result


# Not used anywhere
def transform_to_voltage(x_val, y_min, y_max, x_min, x_max):
    """
    Define a line by two points. Evaluate it at a given point.

    Parameters
    ----------
    x_val: float
        Point at which the line is evaluated.
    y_min : float
        Y-coordinate of first point.
    y_max : float
        Y-coordinate of second point.
    x_min : float
        X-coordinate of first point.
    x_max : float
        X-coordinate of second point.

    Returns
    -------
    (x_val * w) + b : float
        The line is defined by w and b and evaluated at x_val.
    """
    w, b = get_map_to_voltage_vars(y_min, y_max, x_min, x_max)
    return (x_val * w) + b


# Used in utils/transforms.py and
def get_map_to_voltage_vars(y_min, y_max, x_min, x_max):
    """
    Get the scale and the offset of a line between two points.

    Parameters
    ----------
    y_min : float
        Y-coordinate of first point.
    y_max : float
        Y-coordinate of second point.
    x_min : float
        X-coordinate of first point.
    x_max : float
        X-coordinate of second point.

    Returns
    -------
    scale : double
        Scale of the line.
    offset: double
        Offset of the line.
    """
    return get_scale(y_min, y_max, x_min, x_max), get_offset(y_min, y_max, x_min, x_max)


# Only used in this file.
def get_scale(y_min, y_max, x_min, x_max):
    """
    Find the scale/slope of a line defined by points.

    Parameters
    ----------
    y_min : float
        Y-coordinate of first point.
    y_max : float
        Y-coordinate of second point.
    x_min : float
        X-coordinate of first point.
    x_max : float
        X-coordinate of second point.

    Returns
    -------
    (y_min - y_max) / (x_min - x_max) : float

    """
    y = y_min - y_max
    x = x_min - x_max
    return y / x


# Only used in this file.
def get_offset(y_min, y_max, x_min, x_max):
    """
    Get the offset/y-intercept of a line defined by two points.
    Used to transform current data to the input voltage ranges of a device:
    Current range would be (x_min, x_max), voltage range would be (y_min, y_max).
    TODO copy this description to the other functions

    Example
    -------
    Let x_min = 1
        y_min = 1
        x_max = 2
        y_max = 0
    This gives the line defined by the points (1, 1) and (2, 0),
    which is y = 2 - x.
    The function will return the offset, which is 2.
    TODO copy this example to the other functions

    Parameters
    ----------
    y_min : float
        Y-coordinate of first point.
    y_max : float
        Y-coordinate of second point.
    x_min : float
        X-coordinate of first point.
    x_max : float
        X-coordinate of second point.

    Returns
    -------
    ((y_max * x_min) - (y_min * x_max)) / (x_min - x_max) : float
        The offset of the line.

    """
    y = (y_max * x_min) - (y_min * x_max)
    x = x_min - x_max
    return y / x