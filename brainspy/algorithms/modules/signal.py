"""
Set of fitness functions for genetic algorithm and loss functions for gradient
descent.
"""
import warnings

import torch

from brainspy.utils.pytorch import TorchUtils
from brainspy.algorithms.modules.performance.accuracy import get_accuracy


def accuracy_fit(output: torch.Tensor,
                 target: torch.Tensor,
                 default_value=False) -> torch.Tensor:
    """
    Fitness function for genetic algorithm using accuracy of a perceptron.
    Teaches single perceptron to transform output to target and
    evaluates the accuracy; is a percentage.
    Will return default value (0) if indicated.

    Needs at least 10 datapoints.

    Example
    -------
    >>> accuracy_fit(torch.rand((100, 1)), torch.rand(100, 1))
    torch.Tensor(48.)
    >>> accuracy_fit(torch.rand((100, 1)), torch.rand(100, 1), True)
    torch.Tensor(0.0)

    Parameters
    ----------
    output : torch.Tensor
        The output data, shape [n, 1] with n datapoints.
    target : torch.Tensor
        The target data, shape [n, 1] with n datapoints.
    default_value : bool, optional
        Return the default value or not, by default False.

    Returns
    -------
    torch.Tensor
        Default value or calculated fitness.
    """
    if default_value:
        return TorchUtils.format(torch.tensor(0.0))
    else:
        return get_accuracy(output, target)["accuracy_value"]


def corr_fit(output: torch.Tensor,
             target: torch.Tensor,
             default_value=False) -> torch.Tensor:
    """
    Fitness function for genetic algorithm using Pearson correlation.
    See pearsons_correlation for more info.
    Will return default value (-1) if indicated.

    Example
    -------
    >>> corr_fit(torch.rand((100, 1)), torch.rand(100, 1))
    torch.Tensor(0.5)
    >>> corr_fit(torch.rand((100, 1)), torch.rand(100, 1), True)
    torch.Tensor(-1.0)

    Parameters
    ----------
    output : torch.Tensor
        The output data, shape [n, 1] with n datapoints.
    target : torch.Tensor
        The target data, shape [n, 1] with n datapoints.
    default_value : bool, optional
        Return the default value or not, by default False.

    Returns
    -------
    torch.Tensor
        Default value or calculated fitness.
    """
    if default_value:
        return TorchUtils.format(torch.tensor(-1.0))
    else:
        return pearsons_correlation(output, target)


def corrsig_fit(output: torch.Tensor,
                target: torch.Tensor,
                default_value=False) -> torch.Tensor:
    """
    Fitness function for genetic algorithm using correlation and a sigmoid
    function.
    Will return default value (-1) if indicated.

    Note: target data must be binary for this to work.

    Example
    -------
    >>> corrsig_fit(torch.rand((100, 1)), torch.round(torch.rand(100, 1)))
    torch.Tensor(0.5)
    >>> corrsig_fit(torch.rand((100, 1)), torch.round(torch.rand(100, 1)),
                    True)
    torch.Tensor(-1.0)

    Parameters
    ----------
    output : torch.Tensor
        The output data, shape [n, 1] with n datapoints.
    target : torch.Tensor
        The target data, shape [n, 1] with n datapoints;
        should be binary.
    default_value : bool, optional
        Return the default value or not, by default False.

    Returns
    -------
    torch.Tensor
        Default value or calculated fitness.
        Will be NaN if target data is not binary.
    """
    if default_value:
        return TorchUtils.format(torch.tensor(-1.0))
    else:
        corr = pearsons_correlation(output, target)
        sep = output[target == 1].mean() - output[target == 0].mean()
        # average of output where target is 1 minus average of output where
        # target is 0
        sig = torch.sigmoid(-2 * (sep - 2))
        return corr * sig


def pearsons_correlation(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Measure the Pearson correlation between two sets of data (how much the two
    sets are linearly related). Value is between -1 and 1, where 1 is positive
    correlation, -1 is negative, and 0 is no correlation.

    An explanation and the formula for correlation:
    https://www.socscistatistics.com/tests/pearson/

    Example
    -------
    >>> pearsons_correlation(torch.rand((100, 1)), torch.rand((100, 1)))
    torch.Tensor(0.5)

    Parameters
    ----------
    x : torch.Tensor
        Dataset, shape [n, 1] with n datapoints.
    y : torch.Tensor
        Dataset, shape [n, 1] with n datapoints.

    Returns
    -------
    torch.Tensor
        Correlation between x and y (shape []). Will be nan if a data is
        uniform.

    Raises
    ------
    UserWarning
        If result is nan (which happens if a dataset has variance 0, is
        uniform).
    """
    vx = x - x.mean(dim=0)
    vy = y - y.mean(dim=0)
    sum_vx = torch.sum(vx**2)
    sum_vy = torch.sum(vy**2)
    sum_vxy = torch.sum(vx * vy)
    if 0.0 in sum_vx or 0.0 in sum_vy:
        warnings.warn("Variance of dataset is 0, correlation is nan.")
    return sum_vxy / (torch.sqrt(sum_vx) * torch.sqrt(sum_vy))


def corrsig(output: torch.Tensor,
            target: torch.Tensor,
            center: float = 5.0,
            scale: float = 3.0,
            shift: float = 1.1) -> torch.Tensor:
    """
    Loss function for gradient descent using a sigmoid function.

    The default values of the parameters are used for an objective
    function in the Nature Nano paper.

    Example
    -------
    >>> corrsig(torch.rand((100, 1)), torch.round(torch.rand((100, 1))))
    torch.Tensor(2.5)

    Parameters
    ----------
    output : torch.Tensor
        Dataset, shape [n, 1] with n datapoints.
    target : torch.Tensor
        Dataset, shape [n, 1] with n datapoints; should be binary.
    center : float
        Center of the sigmoid.
    scale : float
        Scale of the sigmoid, between 0 and 1.
    shift : float
        Shifting the correlation value.

    Returns
    -------
    torch.Tensor
        Value of loss function.
    """
    # get correlation
    corr = pearsons_correlation(output, target)

    # difference between smallest false negative and largest false positive
    x_high_min = torch.min(output[(target == 1)])
    x_low_max = torch.max(output[(target == 0)])
    delta = x_high_min - x_low_max

    return (shift - corr) / torch.sigmoid((delta - center) / scale)


def fisher_fit(output: torch.Tensor,
               target: torch.Tensor,
               default_value=False) -> torch.Tensor:
    """
    Fitness function for genetic algorithm using Fisher linear discriminant.
    For more information see fisher method.

    Can return default value (0).

    Example
    -------
    >>> fisher_fit(torch.rand((100, 1)), torch.rand((100, 1)),
                   False)
    torch.Tensor(2.5)
    >>> fisher_fit(torch.rand((100, 1)), torch.rand((100, 1)),
                   True)
    torch.Tensor(0.0)

    Parameters
    ----------
    output : torch.Tensor
        The output data, shape [n, 1] with n datapoints.
    target : torch.Tensor
        The target data, shape [n, 1] with n datapoints.
    default_value : bool, optional
        Return the default value or not, by default False.

    Returns
    -------
    torch.Tensor
        Default value or calculated fitness.
    """
    if default_value:
        return TorchUtils.format(torch.tensor(0.0))
    else:
        return fisher(output, target)


def fisher(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Calculate the Fisher linear discriminant between two datasets.
    Used as a loss function for gradient descent.

    More information here:
    https://sthalles.github.io/fisher-linear-discriminant/

    Example
    -------
    >>> fisher(torch.rand((100, 1)), torch.rand((100, 1)))
    torch.Tensor(0.5)

    Parameters
    ----------
    output : torch.Tensor
        The output data, shape [n, 1] with n datapoints.
    target : torch.Tensor
        The target data, shape [n, 1] with n datapoints.

    Returns
    -------
    torch.Tensor
        Value of Fisher linear discriminant.
    """
    x_high = output[(target == 1)]
    x_low = output[(target == 0)]
    m0, m1 = torch.mean(x_low), torch.mean(x_high)
    s0, s1 = torch.var(x_low), torch.var(x_high)
    mean_separation = (m1 - m0)**2
    return mean_separation / (s0 + s1)


def sigmoid_nn_distance(outputs: torch.Tensor,
                        target: torch.Tensor = None,
                        center: float = 0.5,
                        scale: float = 2.0) -> torch.Tensor:
    """
    Sigmoid of nearest neighbour distance: a squashed version of a sum of all
    internal distances between points.
    Used as a loss function for gradient descent.

    The default values of the parameters are used for an objective
    function in the Nature Nano paper.

    Example
    -------
    >>> sigmoid_nn_distance(torch.rand((100, 1)))
    torch.Tensor(20.0)

    Parameters
    ----------
    output : torch.Tensor
        The output data, shape [n, 1] with n datapoints.
    target : torch.Tensor
        The target data, will not be used.
    center : float
        Center of the sigmoid.
    scale : float
        Scale of the sigmoid, between 0 and 1.

    Returns
    -------
    torch.Tensor
        Sigmoid of the sum of the nearest neighbor distances (output).

    Raises
    ------
    UserWarning
        If target data is provided to warn that it will not be used.
    """
    if target is not None:
        warnings.warn(
            "This loss function does not use target values. Target ignored.")
    dist_nn = get_clamped_intervals(outputs, mode="single_nn")
    return -1 * torch.mean(torch.sigmoid(dist_nn / scale) - center)


def get_clamped_intervals(outputs: torch.Tensor,
                          mode: str,
                          boundaries=[0.0, 1.0]) -> torch.Tensor:
    """
    Sort and clamp the data, and find the distances between the datapoints.

    There are three modes:
    "single_nn" - for each point the smaller distance to a neighbor
    "double_nn" - simply the distances between the points
    "intervals" - for each point the summed distance to the point in front
                  and behind it

    Example
    -------
    >>> output = torch.tensor([3.0, 1.0, 8.0, 9.0, 5.0]).unsqueeze(dim=1)
    >>> clamp = [1, 9]
    >>> get_clamped_intervals(output, "single_nn", clamp)
    torch.tensor([0.0, 2.0, 2.0, 1.0, 0.0])
    >>> get_clamped_intervals(output, "double_nn", clamp)
    torch.tensor([0.0, 2.0, 2.0, 3.0, 1.0, 0.0])
    >>> get_clamped_intervals(output, "intervals", clamp)
    torch.tensor([2.0, 4.0, 5.0, 4.0, 1.0])

    Here we have a dataset which ordered is 1, 3, 5, 8, 9.
    The distances between the points are 0, 2, 2, 3, 1, 0 (double).
    The smaller distance for each is 0, 2, 2, 1, 0 (single).
    The sum from both sides is 2, 4, 5, 4, 1 (intervals).

    Parameters
    ----------
    outputs : torch.Tensor
        Dataset, shape [n, 1] with n datapoints.
    mode : str
        Mode for nearest neighbor. Can be
        "single_nn", "double_nn" or "intervals"
    boundaries : list[float], optional
        Boundary values for clamping [min, max].

    Returns
    -------
    torch.Tensor
        Distances between the datapoints.

    Raises
    ------
    UserWarning
        If mode not recognized.
    """
    # First we sort the output, and clip the output to a fixed interval.
    outputs_sorted = outputs.sort(dim=0)[0]
    outputs_clamped = outputs_sorted.clamp(boundaries[0], boundaries[1])

    # Then we prepare two tensors which we subtract from each other to
    # calculate nearest neighbour distances.
    boundaries = TorchUtils.format(boundaries)
    boundary_low = boundaries[0].unsqueeze(0).unsqueeze(1)
    boundary_high = boundaries[1].unsqueeze(0).unsqueeze(1)
    outputs_highside = torch.cat((outputs_clamped, boundary_high), dim=0)
    outputs_lowside = torch.cat((boundary_low, outputs_clamped), dim=0)

    # Most intervals are multiplied by 0.5 because they are shared between two
    # neighbours
    # The first and last interval do not get divided by two because they are
    # not shared
    multiplier = torch.ones_like(outputs_highside)
    multiplier[0] = 1
    multiplier[-1] = 1

    # Calculate the actual distance between points
    dist = (outputs_highside - outputs_lowside) * multiplier

    if mode == "single_nn":
        # Only give nearest neighbour (single!) distance
        dist_nns = torch.cat((dist[1:], dist[:-1]),
                             dim=1)  # both nearest neighbours
        dist_nn = torch.min(dist_nns,
                            dim=1)  # only the closes nearest neighbour
        return dist_nn[0].unsqueeze(
            dim=1)  # entry 0 is the tensor, entry 1 are the indices
    elif mode == "double_nn":
        return dist
    elif mode == "intervals":
        # Determine the intervals between the points, up and down together.
        intervals = dist[1:] + dist[:-1]
        return intervals
    else:
        warnings.warn("Nearest neightbour distance mode not recongized; "
                      "assuming double_nn.")
        return dist
