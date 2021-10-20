from typing import Tuple

import torch


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
    x_val: float,torch.Tensor
        Point at which the line is evaluated.
    y_min : float,torch.Tensor
        Y-coordinate of first point.
    y_max : float,torch.Tensor
        Y-coordinate of second point.
    x_min : float,torch.Tensor
        X-coordinate of first point.
    x_max : float,torch.Tensor
        X-coordinate of second point.

    Returns
    -------
    float or torch.Tensor (sepending on the parameters)
        The line is defined by w and b and evaluated at x_val.

    Raises
    ------
    ZeroDivisionError
        If x_min equals x_max division by 0 occurs.
    """
    if (type(x_max) == int or type(x_max) == float) and (type(x_min) == int or
                                                         type(x_min) == float):
        if x_max < x_min:
            raise AssertionError("x_min cannot be greater than x_max")

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
    y_min : float,torch.Tensor
        Y-coordinate of first point. In a current to voltage linear transformation,
        the expected minimum value(s) for the voltage.
    y_max : float,torch.Tensor
        Y-coordinate of second point. In a current to voltage linear transformation,
        the expected maximum value(s) for the voltage.
    x_min : float,torch.Tensor
        X-coordinate of first point. In a current to voltage linear transformation,
        the expected minimum value(s) for the current.
    x_max : float,torch.Tensor
        X-coordinate of second point. In a current to voltage linear transformation,
        the expected minimum value(s) for the current.

    Returns
    -------
    scale : double - if parameters are of type - float
            torch.Tensor - if parameters are of type - torch.Tensor
        Scale of the line.
    offset: double - if parameters are of type - float
            torch.Tensor - if parameters are of type - torch.Tensor
        Offset of the line.

    Raises
    ------
    ZeroDivisionError
        If x_min equals x_max division by 0 occurs.
    """
    if (type(x_max) == int or type(x_max) == float) and (type(x_min) == int or
                                                         type(x_min) == float):
        if x_max < x_min:
            raise AssertionError("x_min cannot be greater than x_max")
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
    y_min : float,torch.Tensor
        Y-coordinate of first point. In a current to voltage linear transformation,
        the expected minimum value(s) for the voltage.
    y_max : float,torch.Tensor
        Y-coordinate of second point. In a current to voltage linear transformation,
        the expected maximum value(s) for the voltage.
    x_min : float,torch.Tensor
        X-coordinate of first point. In a current to voltage linear transformation,
        the expected minimum value(s) for the current.
    x_max : float,torch.Tensor
        X-coordinate of second point. In a current to voltage linear transformation,
        the expected minimum value(s) for the current.

    Returns
    -------
    double - if parameters are of type - float
    torch.Tensor - if parameters are of type - torch.Tensor
        The scale of the line.

    Raises
    ------
    ZeroDivisionError
        If x_min equals x_max division by 0 occurs.
    """
    if (type(x_max) == int or type(x_max) == float) and (type(x_min) == int or
                                                         type(x_min) == float):
        if x_max < x_min:
            raise AssertionError("x_min cannot be greater than x_max")
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
    y_min : float,torch.Tensor
        Y-coordinate of first point. In a current to voltage linear transformation,
        the expected minimum value(s) for the voltage.
    y_max : float,torch.Tensor
        Y-coordinate of second point. In a current to voltage linear transformation,
        the expected maximum value(s) for the voltage.
    x_min : float,torch.Tensor
        X-coordinate of first point. In a current to voltage linear transformation,
        the expected minimum value(s) for the current.
    x_max : float,Torch.Tensor
        X-coordinate of second point. In a current to voltage linear transformation,
        the expected minimum value(s) for the current.

    Returns
    -------
    double - if parameters are of type - float
    torch.Tensor - if parameters are of type - torch.Tensor
        The offset of the line.

    Raises
    ------
    ZeroDivisionError
        If x_min equals x_max division by 0 occurs.
    """
    if (type(x_max) == int or type(x_max) == float) and (type(x_min) == int or
                                                         type(x_min) == float):
        if x_max < x_min:
            raise AssertionError("x_min cannot be greater than x_max")
    return ((y_max * x_min) - (y_min * x_max)) / (x_min - x_max)


# Used in conv.py.
def format_input_ranges(input_min: float, input_max: float,
                        output_ranges: torch.Tensor):
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
