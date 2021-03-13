from typing import Tuple, Sequence
import torch
from brainspy.utils.pytorch import TorchUtils
# Moved most transforms to bspytasks.

# Used in bn.py
class CurrentToVoltage:
    """
    Class that uses a linear function to transform current to voltage.
    """
    def __init__(self, current_range: Sequence[float], voltage_range: Sequence[float], cut=True):
        """
        Initialize object, find linear transform parameters for each current-voltage pair.

        Example
        -------
        >>> CurrentToVoltage([[0, 1], [1, 2]], [[1, 2], [1, 0]])

        This example defines two transformations, the first with current range 0 to 1 and voltage
        range 1 to 2, the second with current range 1 to 2 and voltage range 1 to 0.

        Parameters
        ----------
        current_range : Sequence[float]
            The data for the current range.
            [[current1_min, current1_max], [current2_min, current2_max], ...]
        voltage_range : Sequence[float]
            The data for the voltage range.
            [[current1_min, current1_max], [current2_min, current2_max], ...]
        cut : bool, optional
            Indicate whether to use cut (torch.clamp) when calling the object, by default True.
            If set to true, input values outside of the current range will be set to either the
            minimum or maximum, depending on whether the value is below or above the range.

        Raises
        ------
        Exception
            If the current and voltage ranges are different in length.
        """
        if len(current_range) != len(voltage_range):
            raise Exception("Mapping ranges are different in length")

        # Determine the transform parameters for each pair.
        self.map_variables = TorchUtils.get_tensor_from_list(
            [
                transform_current_to_voltage(
                    voltage_range[i][0],
                    voltage_range[i][1],
                    current_range[i][0],
                    current_range[i][1],
                )
                for i in range(len(current_range))
            ]
        )
        self.current_range = current_range
        self.cut = cut

    def __call__(self, x_value: torch.Tensor) -> torch.Tensor:
        """
        For given input currents determine the output voltages using the linear transforms.

        Example
        -------
        >>> ctv = CurrentToVoltage([[0, 1], [1, 2]], [[1, 2], [1, 0]])
        >>> ctv([1, 1])
        [2, 1]

        This example defines two transformations, the first with current range 0 to 1 and voltage
        range 1 to 2, the second with current range 1 to 2 and voltage range 1 to 0.
        With the input values 1 for the first transform and 1 for the second we get the outputs
        2 and 1 respectively.

        Parameters
        ----------
        x_value : torch.Tensor
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
        # If cut will be applied, we need a copy of the x values to apply it to.
        x_copy = x_value.clone()
        result = torch.zeros_like(x_value)

        if not (len(x_value.shape) == 2 and x_value.shape[1] == len(self.map_variables)):
            raise Exception("Input shape not supported.")

        for i in range(len(self.map_variables)):
            if self.cut:
                x_copy[:, i] = torch.clamp(
                    x_value[:, i], min=self.current_range[i][0], max=self.current_range[i][1]
                )
            result[:, i] = (x_copy[:, i] * self.map_variables[i][0]) + self.map_variables[i][1]

        return result


# Not used anywhere
def transform_to_voltage(
    y_min: float, y_max: float, x_min: float, x_max: float, x_val: float
) -> float:
    """
    Define a line by two points. Evaluate it at a given point.
    Used to transform current data to the input voltage ranges of a device:
    Current range would be (x_min, x_max), voltage range would be (y_min, y_max).

    Example
    -------
    >>> transform_to_voltage(x_min=1, y_min=1, x_max=2, y_max=0, x_val=1)
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
    scale, offset = transform_current_to_voltage(y_min, y_max, x_min, x_max)
    return (x_val * scale) + offset


# Only used here.
def transform_current_to_voltage(
    y_min: float, y_max: float, x_min: float, x_max: float
) -> Tuple[float, float]:
    """
    Get the scale and offset of a line defined by two points.
    Used to transform current data to the input voltage ranges of a device:
    Current range would be (x_min, x_max), voltage range would be (y_min, y_max).

    Example
    -------
    >>> transform_current_to_voltage(x_min=1, y_min=1, x_max=2, y_max=0)
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
