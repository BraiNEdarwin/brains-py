import torch
import numpy as np

from torch import Tensor
from typing import Sequence, Tuple, Union

from brainspy.utils.pytorch import TorchUtils

# Used in processors/processor.py.
def merge_electrode_data(
    inputs,
    control_voltages,
    input_indices: Sequence[int],
    control_voltage_indices,
    use_torch=True,
) -> Union[np.array, Tensor]:
    """
    Merge data from two electrodes with the specified indices for each.
    Need to indicate whether numpy or torch is used. The result will
    have the same type as the input.

    Example
    -------
    >>> inputs = np.array([[1.0, 3.0], [2.0, 4.0]])
    >>> control_voltages = np.array([[5.0, 7.0], [6.0, 8.0]])
    >>> input_indices = [0, 2]
    >>> control_voltage_indices = [3, 1]
    >>> electrodes.merge_electrode_data(
    ...     inputs=inputs,
    ...     control_voltages=control_voltages,
    ...     input_indices=input_indices,
    ...     control_voltage_indices=control_voltage_indices,
    ...     use_torch=False,
    ... )
    np.array([[1.0, 7.0, 3.0, 5.0], [2.0, 8.0, 4.0, 6.0]])

    Merging two arrays of size 2x2, resulting in an array of size 2x4.

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
def linear_transform(
    y_min: float, y_max: float, x_min: float, x_max: float, x_val: float
) -> float:
    """
    Applies a linear transformation of a point within a range, to a point within another range.
    Used to transform current data to the input voltage ranges of a device:
    Current range would be (x_min, x_max), voltage range would be (y_min, y_max).

    Example
    -------
    >>> linear_transform(x_min=1, y_min=1, x_max=2, y_max=0, x_val=1)
    1

    This gives the line defined by the points (1, 1) and (2, 0),
    which is y = 2 - x.
    The function will return the line evaluated at x = 1, which is 1.

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
    (x_val * scale) + offset : float
        The line is defined by w and b and evaluated at x_val.

    Raises
    ------
    ZeroDivisionError
        If x_min equals x_max division by 0 occurs.
    """
    scale, offset = get_linear_transform_constants(y_min, y_max, x_min, x_max)
    return (x_val * scale) + offset


# Used in utils/transforms.py and
def get_linear_transform_constants(
    y_min: float, y_max: float, x_min: float, x_max: float
) -> Tuple[float, float]:
    """
    Get the scale and offset constants of a line defined by two points.
    The constants can be used to apply a linear transformation of a point 
    within a range, to a point within another range.
    Used to transform current data to the input voltage ranges of a device:
    Current range would be (x_min, x_max), voltage range would be (y_min, y_max).

    Example
    -------
    >>> get_linear_transform_constants(x_min=1, y_min=1, x_max=2, y_max=0)
    (-1, 2)

    This gives the line defined by the points (1, 1) and (2, 0),
    which is y = 2 - x.
    The function will return the scale and offset, which are -1 and 2 respectively.

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

    Raises
    ------
    ZeroDivisionError
        If x_min equals x_max division by 0 occurs.
    """
    return get_scale(y_min, y_max, x_min, x_max), get_offset(y_min, y_max, x_min, x_max)


# Only used in this file.
def get_scale(y_min: float, y_max: float, x_min: float, x_max: float) -> float:
    """
    Get the scale/slope of a line defined by two points.
    Used to transform current data to the input voltage ranges of a device:
    Current range would be (x_min, x_max), voltage range would be (y_min, y_max).

    Example
    -------
    >>> get_scale(x_min=1, y_min=1, x_max=2, y_max=0)
    -1

    This gives the line defined by the points (1, 1) and (2, 0),
    which is y = 2 - x.
    The function will return the scale, which is -1.

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

    Raises
    ------
    ZeroDivisionError
        If x_min equals x_max division by 0 occurs.
    """
    return (y_min - y_max) / (x_min - x_max)


# Only used in this file.
def get_offset(y_min: float, y_max: float, x_min: float, x_max: float) -> float:
    """
    Get the offset/y-intercept of a line defined by two points.
    Used to transform current data to the input voltage ranges of a device:
    Current range would be (x_min, x_max), voltage range would be (y_min, y_max).

    Example
    -------
    >>> get_offset(x_min=1, y_min=1, x_max=2, y_max=0)
    2

    This gives the line defined by the points (1, 1) and (2, 0),
    which is y = 2 - x.
    The function will return the offset, which is 2.

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

    Raises
    ------
    ZeroDivisionError
        If x_min equals x_max division by 0 occurs.
    """
    return ((y_max * x_min) - (y_min * x_max)) / (x_min - x_max)

def format_input_ranges(input_min, input_max, output_ranges):
    input_ranges = torch.ones_like(output_ranges)
    input_ranges[0] *= input_min
    input_ranges[1] *= input_max
    return input_ranges
