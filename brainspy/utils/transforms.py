"""
Class for transforming from current to voltage using linear transformations.
The main class CurrentToVoltage takes arrays of currents and voltages.
It then calculates the linear transform (scale/slope and offset/intercept) for
each current-voltage pair by finding the line between two points.
These transforms are used to map the data to the inputs of the DNPU.

The following link gives more information on linear functions:
https://en.wikipedia.org/wiki/Linear_function_(calculus)
"""

from typing import Tuple, Sequence

import torch

from brainspy.utils.pytorch import TorchUtils


class CurrentToVoltage:
    """
    Class that uses a linear function to transform current to voltage for sets
    of points.

    Attributes
    ----------
    map_variables : Sequence[Sequence[float]]
        The linear map parameters (scale and offset) of each pair of data
        points.
    current_range : Sequence[Sequence[float]]
        The current values.
    cut : bool
        Indicate whether to apply cut to the output.
    """
    def __init__(
        self,
        current_range: Sequence[Sequence[float]],
        voltage_range: Sequence[Sequence[float]],
    ):
        """
        Initialize object, find linear transform parameters for each
        current-voltage pair.

        Example
        -------
        >>> CurrentToVoltage([[0, 1], [1, 2]], [[1, 2], [1, 0]])

        This example defines two transformations, the first with current range
        0 to 1 and voltage range 1 to 2, the second with current range 1 to 2
        and voltage range 1 to 0.

        Parameters
        ----------
        current_range : Sequence[float]
            The data for the current range.
            [[current1_min, current1_max], [current2_min, current2_max], ...]
        voltage_range : Sequence[float]
            The data for the voltage range.
            [[current1_min, current1_max], [current2_min, current2_max], ...]
        cut : bool, optional
            Indicate whether to use cut (torch.clamp) when calling the object,
            by default True. If set to true, input values outside of the
            current range will be set to either the minimum or maximum,
            depending on whether the value is below or above the range.

        Raises
        ------
        Exception
            If the current and voltage ranges are different in length.
        """
        if len(current_range) != len(voltage_range):
            raise Exception("Mapping ranges are different in length")

        self.scale, self.offset = get_linear_transform_constants(voltage_range.T[0].T, voltage_range.T[1].T, current_range.T[0].T, current_range.T[1].T)
        self.current_range = current_range

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        For given input currents determine the output voltages using the
        linear transforms.

        Example
        -------
        >>> ctv = CurrentToVoltage([[0, 1], [1, 2]], [[1, 2], [1, 0]])
        >>> ctv([1, 1])
        [2, 1]

        This example defines two transformations, the first with current range
        0 to 1 and voltage range 1 to 2, the second with current range 1 to 2
        and voltage range 1 to 0. With the input values 1 for the first
        transform and 1 for the second we get the outputs 2 and 1 respectively.

        Parameters
        ----------
        x : torch.Tensor
            Input current values.

        Returns
        -------
        result : torch.Tensor
            Output voltage values.

        Raises
        ------
        Exception
            If the shape the dimension of the input is wrong.
        """
        return (self.scale * x) + self.offset


# Not used anywhere
def linear_transform(y_min: float, y_max: float, x_min: float, x_max: float,
                     x_val: float) -> float:
    """
    Define a line by two points. Evaluate it at a given point.
    Used to transform current data to the input voltage ranges of a device:
    Current range would be (x_min, x_max), voltage range would be
    (y_min, y_max).

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
    float
        The line is defined by w and b and evaluated at x_val.

    Raises
    ------
    ZeroDivisionError
        If x_min equals x_max division by 0 occurs.
    """
    scale, offset = get_linear_transform_constants(y_min, y_max, x_min, x_max)
    return (x_val * scale) + offset


# Only used here.
def get_linear_transform_constants(y_min: float, y_max: float, x_min: float,
                                   x_max: float) -> Tuple[float, float]:
    """
    Get the scale and offset of a line defined by two points.
    Used to transform current data to the input voltage ranges of a device:
    Current range would be (x_min, x_max), voltage range would be
    (y_min, y_max).

    Example
    -------
    >>> get_linear_transform_constants(x_min=1, y_min=1, x_max=2, y_max=0)
    (-1, 2)

    This gives the line defined by the points (1, 1) and (2, 0),
    which is y = 2 - x.
    The function will return the scale and offset, which are -1 and 2
    respectively.

    Parameters
    ----------
    y_min : float
        Y-coordinate of first point. In a current to voltage linear transformation,
        the expected minimum value(s) for the voltage.
    y_max : float
        Y-coordinate of second point. In a current to voltage linear transformation,
        the expected maximum value(s) for the voltage.
    x_min : float
        X-coordinate of first point. In a current to voltage linear transformation,
        the expected minimum value(s) for the current.
    x_max : float
        X-coordinate of second point. In a current to voltage linear transformation,
        the expected minimum value(s) for the current.

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
    return get_scale(y_min, y_max, x_min,
                     x_max), get_offset(y_min, y_max, x_min, x_max)


# Only used in this file.
def get_scale(y_min: float, y_max: float, x_min: float, x_max: float) -> float:
    """
    Get the scale/slope of a line defined by two points.
    Used to transform current data to the input voltage ranges of a device:
    Current range would be (x_min, x_max), voltage range would be
    (y_min, y_max).

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
        Y-coordinate of first point. In a current to voltage linear transformation,
        the expected minimum value(s) for the voltage.
    y_max : float
        Y-coordinate of second point. In a current to voltage linear transformation,
        the expected maximum value(s) for the voltage.
    x_min : float
        X-coordinate of first point. In a current to voltage linear transformation,
        the expected minimum value(s) for the current.
    x_max : float
        X-coordinate of second point. In a current to voltage linear transformation,
        the expected minimum value(s) for the current.

    Returns
    -------
    float
        The scale of the line.

    Raises
    ------
    ZeroDivisionError
        If x_min equals x_max division by 0 occurs.
    """
    return (y_min - y_max) / (x_min - x_max)


# Only used in this file.
def get_offset(y_min: float, y_max: float, x_min: float,
               x_max: float) -> float:
    """
    Get the offset/y-intercept of a line defined by two points.
    Used to transform current data to the input voltage ranges of a device:
    Current range would be (x_min, x_max), voltage range would be
    (y_min, y_max).

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
        Y-coordinate of first point. In a current to voltage linear transformation,
        the expected minimum value(s) for the voltage.
    y_max : float
        Y-coordinate of second point. In a current to voltage linear transformation,
        the expected maximum value(s) for the voltage.
    x_min : float
        X-coordinate of first point. In a current to voltage linear transformation,
        the expected minimum value(s) for the current.
    x_max : float
        X-coordinate of second point. In a current to voltage linear transformation,
        the expected minimum value(s) for the current.

    Returns
    -------
    float
        The offset of the line.

    Raises
    ------
    ZeroDivisionError
        If x_min equals x_max division by 0 occurs.
    """
    return ((y_max * x_min) - (y_min * x_max)) / (x_min - x_max)


# Used in conv.py.
def format_input_ranges(input_min, input_max, output_ranges):
    """
    Generate a tensor of input ranges with the same shape as given output
    ranges. All the input ranges are from input_min to input_max.

    Example
    -------
    >>> t = tensor([[1, 2, 3], [4, 5, 6]])
    >>> format_input_ranges(7, 8, t)
    torch.tensor([[7, 7, 7], [8, 8, 8]])

    Parameters
    ----------
    input_min : float
        Minimum value for input.
    input_max : float
        Maximum value for input.
    output_ranges : torch.Tensor
        Tensor of output ranges.

    Returns
    -------
    torch.Tensor
        A tensor with the same shape as output_ranges with the values of
        input_min and input_max.
    """
    input_ranges = torch.ones_like(output_ranges)
    input_ranges[0] *= input_min
    input_ranges[1] *= input_max
    return input_ranges
