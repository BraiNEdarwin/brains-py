import torch
from typing import Tuple


def linear_transform(y_min, y_max, x_min, x_max, x_val):
    """
    Define a line by two points. Evaluate it at a given point.
    The points are expressed as float values or as torch tensors.
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
    float or torch.Tensor (depending on the type of the parameters)
        The line is defined by w and b and evaluated at x_val.

    Raises
    ------
    ZeroDivisionError
        If x_min equals x_max division by 0 occurs.
    """

    scale, offset = get_linear_transform_constants(y_min, y_max, x_min, x_max)
    return (x_val * scale) + offset


def get_linear_transform_constants(
        y_min: torch.Tensor, y_max: torch.Tensor, x_min: torch.Tensor,
        x_max: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get the scale and offset of a line defined by two points.
    The two points are expressed as torch tensors.
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

    >>> get_linear_transform_constants(x_min=torch.tensor([[1, 4]]),
                                       y_min=torch.tensor([[16, 14]]),
                                       x_max=torrch.tensor([[7, 4]]),
                                       y_max=torch.tensor([[12, 14]]))
    (tensor([[0.3333, 1.0000]]), tensor([[6.6667, -0.0000]]))

    Parameters
    ----------
    y_min : torch.Tensor
        Y-coordinate of first point. In a current to voltage linear transformation,
        the expected minimum value(s) for the voltage.
    y_max : torch.Tensor
        Y-coordinate of second point. In a current to voltage linear transformation,
        the expected maximum value(s) for the voltage.
    x_min : torch.Tensor
        X-coordinate of first point. In a current to voltage linear transformation,
        the expected minimum value(s) for the current.
    x_max : torch.Tensor
        X-coordinate of second point. In a current to voltage linear transformation,
        the expected minimum value(s) for the current.

    Returns
    -------
    [scale,offset]
    scale : torch.Tensor
        Scale of the line.
    offset: torch.Tensor
        Offset of the line.

    Raises
    ------
    ZeroDivisionError
        If x_min equals x_max division by 0 occurs.
    """
    return get_scale(y_min, y_max, x_min,
                     x_max), get_offset(y_min, y_max, x_min, x_max)


def get_scale(y_min: torch.Tensor, y_max: torch.Tensor, x_min: torch.Tensor,
              x_max: torch.Tensor) -> torch.Tensor:
    """
    Get the scale/slope of a line defined by two points.
    The points are expressed as torch tensors.
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

    >>> get_scale(x_min=torch.tensor([[1, 4]]),
                  y_min=torch.tensor([[16, 14]]),
                  x_max=torrch.tensor([[7, 4]]),
                  y_max=torch.tensor([[12, 14]]))
    tensor([[0.3333, 1.0000]])

    Parameters
    ----------
    y_min : torch.Tensor
        Y-coordinate of first point. In a current to voltage linear transformation,
        the expected minimum value(s) for the voltage.
    y_max : torch.Tensor
        Y-coordinate of second point. In a current to voltage linear transformation,
        the expected maximum value(s) for the voltage.
    x_min : torch.Tensor
        X-coordinate of first point. In a current to voltage linear transformation,
        the expected minimum value(s) for the current.
    x_max : torch.Tensor
        X-coordinate of second point. In a current to voltage linear transformation,
        the expected minimum value(s) for the current.

    Returns
    -------
    torch.Tensor
        The scale of the line.

    Raises
    ------
    ZeroDivisionError
        If x_min equals x_max division by 0 occurs.
    """
    check_values(x_max, x_min, y_max, y_min)
    return (y_min - y_max) / (x_min - x_max)


def get_offset(y_min: torch.Tensor, y_max: torch.Tensor, x_min: torch.Tensor,
               x_max: torch.Tensor) -> torch.Tensor:
    """
    Get the offset/y-intercept of a line defined by two points.
    The points are expressed as torch tensors.
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

    >>> get_offset(x_min=torch.tensor([[1, 4]]),
                   y_min=torch.tensor([[16, 14]]),
                   x_max=torrch.tensor([[7, 4]]),
                   y_max=torch.tensor([[12, 14]]))
    tensor([[6.6667, -0.0000]])

    Parameters
    ----------
    y_min : torch.Tensor
        Y-coordinate of first point. In a current to voltage linear transformation,
        the expected minimum value(s) for the voltage.
    y_max : torch.Tensor
        Y-coordinate of second point. In a current to voltage linear transformation,
        the expected maximum value(s) for the voltage.
    x_min : torch.Tensor
        X-coordinate of first point. In a current to voltage linear transformation,
        the expected minimum value(s) for the current.
    x_max : Torch.Tensor
        X-coordinate of second point. In a current to voltage linear transformation,
        the expected minimum value(s) for the current.

    Returns
    -------
    torch.Tensor
        The offset of the line.

    Raises
    ------
    ZeroDivisionError
        If x_min equals x_max division by 0 occurs.
    """
    check_values(x_max, x_min, y_max, y_min)
    return ((y_max * x_min) - (y_min * x_max)) / (x_min - x_max)


def check_values(x_max, x_min, y_max, y_min):
    """
    To check wheather the arguments provided to the functions - get_offset and get_scale
    are valid by raising an Assertion Error if x_max < x_min or y_max < y_min

    Parameters
    ----------
    y_min : torch.Tensor
        Y-coordinate of first point. In a current to voltage linear transformation,
        the expected minimum value(s) for the voltage.
    y_max : torch.Tensor
        Y-coordinate of second point. In a current to voltage linear transformation,
        the expected maximum value(s) for the voltage.
    x_min : torch.Tensor
        X-coordinate of first point. In a current to voltage linear transformation,
        the expected minimum value(s) for the current.
    x_max : Torch.Tensor
        X-coordinate of second point. In a current to voltage linear transformation,
        the expected minimum value(s) for the current.

    Raises
    ------
    AssertionError
        raised if x_max < x_min when the type of the arguments are int or float
    AssertionError
        raised if y_max < y_min when the type of the arguments are int or float
    AssertionError
        raised if x_max < x_min when the arguments are provided as torch tensors
    AssertionError
         raised if y_max < y_min when the arguments are provided as torch tensors
    """
    if (type(x_max) == int or type(x_max) == float) and (type(x_min) == int or
                                                         type(x_min) == float):
        assert x_max >= x_min, "x_min cannot be greater than x_max"
        assert (type(y_max) == int or type(y_max) == float) and (
            type(y_min) == int or type(y_min) == float
        ), "x_max and x_min should be of the same datatype as y_max and y_min"
        assert (y_max >= y_min), "y_min cannot be greater than y_max"
    else:
        assert type(x_min) == torch.Tensor and type(
            x_max
        ) == torch.Tensor, "x_min and x_max should be either integer, floats or torch.Tensor"
        assert type(y_min) == torch.Tensor and type(
            y_max
        ) == torch.Tensor, "y_min and y_max should be either integer, floats or torch.Tensor"
        assert (x_max >=
                x_min).all().item(), "x_min cannot be greater than x_max"
        assert (y_max >=
                y_min).all().item(), "y_min cannot be greater than y_max"
