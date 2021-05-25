"""
Set of functions to measure separability and similarity of signals.
"""
import warnings
from typing import Union

import torch

from brainspy.algorithms.modules.performance.accuracy import get_accuracy

# TODO: implement corr_lin_fit (AF's last fitness function)? Is this relevant?


def accuracy_fit(output: torch.Tensor,
                 target: torch.Tensor,
                 default_value=False) -> Union[float, torch.Tensor]:
    """
    Measure the separability of a fit or return a default value (0):
    Teaches single perceptron to transform output to target and
    evaluates the accuracy; is a percentage.

    Example
    -------
    >>> accuracy_fit(torch.rand((100, 1)), torch.rand(100, 1))
    torch.Tensor(48.)
    >>> accuracy_fit(torch.rand((100, 1)), torch.rand(100, 1), True)
    0.0

    This example shows usage of both options of the method with random
    tensors.

    Parameters
    ----------
    output : torch.Tensor
        The output signal, should be n-by-1 with n datapoints.
    target : torch.Tensor
        The target signal, should be n-by-1 with n datapoints.
    default_value : bool, optional
        Return the default value or not, by default False.

    Returns
    -------
    float or torch.Tensor
        Default value or tensor with accuracy percentage.
    """
    if default_value:
        return 0.0
    else:
        return get_accuracy(output, target)["accuracy_value"]


def corr_fit(output: torch.Tensor,
             target: torch.Tensor,
             default_value=False) -> Union[float, torch.Tensor]:
    """
    Measure the similarity of two signals using pearson correlation or return
    default value (-1).
    See pearson_correlation method documentation for more info.

    Example
    -------
    >>> corr_fit(torch.rand((100, 1)), torch.rand(100, 1))
    torch.Tensor(0.5)
    >>> corr_fit(torch.rand((100, 1)), torch.rand(100, 1), True)
    -1.0

    This example shows usage of both options of the method with random
    tensors.

    Parameters
    ----------
    output : torch.Tensor
        The output signal, should be n-by-1 with n datapoints.
    target : torch.Tensor
        The target signal, should be n-by-1 with n datapoints.
    default_value : bool, optional
        Return the default value or not, by default False.

    Returns
    -------
    float or torch.Tensor
        Default value or tensor with correlation.
    """
    if default_value:
        return -1.0
    else:
        return pearsons_correlation(output[:, 0], target[:, 0])


def corrsig_fit(output: torch.Tensor,
                target: torch.Tensor,
                default_value=False) -> Union[float, torch.Tensor]:
    """
    Measure the similarity of two signals using a combination of a sigmoid
    with pre-defined separation threshold and the correlation function, or
    return default value (-1).

    Note: target signal must be binary for this to work.

    Example
    -------
    >>> corrsig_fit(torch.rand((100, 1)), torch.roundtorch.rand(100, 1)))
    torch.Tensor(0.5)
    >>> corrsig_fit(torch.rand((100, 1)), torch.round(torch.rand(100, 1)),
                    True)
    -1.0

    This example shows usage of both options of the method with random
    tensors.

    Parameters
    ----------
    output : torch.Tensor
        The output signal, should be n-by-1 with n datapoints.
    target : torch.Tensor
        The target signal, should be n-by-1 with n datapoints;
        should be binary.
    default_value : bool, optional
        Return the default value or not, by default False.

    Returns
    -------
    float or torch.Tensor
        Default value or tensor with correlation.
        Will be NaN if target signal is not binary.
    """
    if default_value:
        return -1.0
    else:
        corr = pearsons_correlation(output[:, 0], target[:, 0])
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
    >>> corrsig_fit(torch.rand(100), torch.rand(100))
    torch.Tensor(0.5)

    Parameters
    ----------
    x : torch.Tensor
        Dataset, should be one dimensional.
    y : torch.Tensor
        Dataset, should be one dimensional.

    Returns
    -------
    torch.Tensor
        Correlation between x and y (shape []). Will be nan if a signal is
        uniform.
    """
    vx = x - x.mean(dim=0)
    vy = y - y.mean(dim=0)
    return torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx**2)) *
                                 torch.sqrt(torch.sum(vy**2)))


def corrsig(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Measures similarity of two signals using a predefined sigmoid function.

    Example
    -------
    >>> corrsig_fit(torch.rand(100), torch.ones_like(torch.rand(100)))
    torch.Tensor(2.5)

    Parameters
    ----------
    x : torch.Tensor
        Dataset, should be one dimensional.
    y : torch.Tensor
        Dataset, should be one dimensional.

    Returns
    -------
    torch.Tensor
        Similarity (shape []).
    """
    # get correlation
    corr = torch.mean(
        (output - torch.mean(output)) * (target - torch.mean(target))) / (
            torch.std(output) * torch.std(target) + 1e-10)

    # difference between smallest false negative and largest false positive
    x_high_min = torch.min(output[(target == 1)])
    x_low_max = torch.max(output[(target == 0)])
    delta = x_high_min - x_low_max

    return (1.1 - corr) / torch.sigmoid((delta - 5) / 3)


def sqrt_corrsig(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Measures similarity of two signals using a predefined sigmoid function.

    Parameters
    ----------
    x : torch.Tensor
        Dataset, should be one dimensional.
    y : torch.Tensor
        Dataset, should be one dimensional.

    Returns
    -------
    torch.Tensor
        Similarity (shape []).
    """
    # get correlation
    corr = torch.mean(
        (output - torch.mean(output)) * (target - torch.mean(target))) / (
            torch.std(output) * torch.std(target) + 1e-10)

    # difference between smallest false negative and largest false positive
    x_high_min = torch.min(output[(target == 1)])
    x_low_max = torch.max(output[(target == 0)])
    delta = x_high_min - x_low_max

    return (1.0 - corr)**(1 / 2) / torch.sigmoid((delta - 2) / 5)


def fisher_fit(output: torch.Tensor,
               target: torch.Tensor,
               default_value=False) -> Union[float, torch.Tensor]:
    """
    Apply a fisher fit? or return default value (0)
    TODO description

    Parameters
    ----------
    TODO check dimensions
    output : torch.Tensor
        The output signal, should be n-by-1 with n datapoints.
    target : torch.Tensor
        The target signal, should be n-by-1 with n datapoints.
    default_value : bool, optional
        Return the default value or not, by default False.

    Returns
    -------
    float or torch.Tensor
        Default value or tensor with fisher value.
        TODO return
    """
    if default_value:
        return 0
    else:
        return -fisher(output, target)


def fisher(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Separates classes irrespective of assignments.
    Reliable, but insensitive to actual classes
    TODO description

    Parameters
    ----------
    TODO check dimensions
    output : torch.Tensor
        The output signal, should be n-by-1 with n datapoints.
    target : torch.Tensor
        The target signal, should be n-by-1 with n datapoints.

    Returns
    -------
    torch.Tensor
        [description] TODO return
    """
    x_high = output[(target == 1)]
    x_low = output[(target == 0)]
    m0, m1 = torch.mean(x_low), torch.mean(x_high)
    s0, s1 = torch.var(x_low), torch.var(x_high)
    mean_separation = (m1 - m0)**2
    return -mean_separation / (s0 + s1)


def fisher_added_corr(output: torch.Tensor,
                      target: torch.Tensor) -> torch.Tensor:
    """
    Fisher and correlation, added together. TODO description

    Parameters
    ----------
    TODO check dimensions
    output : torch.Tensor
        The output signal, should be n-by-1 with n datapoints.
    target : torch.Tensor
        The target signal, should be n-by-1 with n datapoints.

    Returns
    -------
    torch.Tensor
        [description] TODO return
    """
    x_high = output[(target == 1)]
    x_low = output[(target == 0)]
    m0, m1 = torch.mean(x_low), torch.mean(x_high)
    s0, s1 = torch.var(x_low), torch.var(x_high)
    mean_separation = (m1 - m0)**2
    corr = torch.mean(
        (output - torch.mean(output)) * (target - torch.mean(target))) / (
            torch.std(output) * torch.std(target) + 1e-10)
    return (1 - corr) - 0.5 * mean_separation / (s0 + s1)


def fisher_multipled_corr(output: torch.Tensor,
                          target: torch.Tensor) -> torch.Tensor:
    """
    Fisher and correlation, multiplied together. TODO description

    Parameters
    ----------
    TODO check dimensions
    output : torch.Tensor
        The output signal, should be n-by-1 with n datapoints.
    target : torch.Tensor
        The target signal, should be n-by-1 with n datapoints.

    Returns
    -------
    torch.Tensor
        [description] TODO return
    """
    x_high = output[(target == 1)]
    x_low = output[(target == 0)]
    m0, m1 = torch.mean(x_low), torch.mean(x_high)
    s0, s1 = torch.var(x_low), torch.var(x_high)
    mean_separation = (m1 - m0)**2
    corr = torch.mean(
        (output - torch.mean(output)) * (target - torch.mean(target))) / (
            torch.std(output) * torch.std(target) + 1e-10)
    return (1 - corr) * (s0 + s1) / mean_separation


def sigmoid_nn_distance(outputs: torch.Tensor,
                        target: torch.Tensor = None) -> torch.Tensor:
    """
    Sigmoid nearest neighbour distance: a squeshed version of a sum of all
    internal distances between points.
    # TODO description

    Parameters
    ----------
    TODO check dimensions
    output : torch.Tensor
        The output signal, should be n-by-1 with n datapoints.
    target : torch.Tensor
        The target signal, will not be used.

    Returns
    -------
    torch.Tensor
        [description] # TODO return

    Raises
    ------
    UserWarning
        If target data is provided to warn that it will not be used.
    """
    if target is not None:
        warnings.warn(
            "This loss function does not use target values. Target ignored.")
    dist_nn = get_clamped_intervals(outputs, mode="single_nn")
    return -1 * torch.mean(torch.sigmoid(dist_nn / 2) - 0.5)


def get_clamped_intervals(outputs: torch.Tensor, mode, boundaries=[-352, 77]):
    """[summary]
    TODO entire docstring

    Parameters
    ----------
    outputs : torch.Tensor
        [description]
    mode : [type]
        [description]
    boundaries : list, optional
        [description], by default [-352, 77]

    Returns
    -------
    [type]
        [description]
    """
    # First we sort the output, and clip the output to a fixed interval.
    outputs_sorted = outputs.sort(dim=0)[0]
    outputs_clamped = outputs_sorted.clamp(boundaries[0], boundaries[1])

    # THen we prepare two tensors which we subtract from each other to
    # calculate nearest neighbour distances.
    boundaries = torch.tensor(boundaries,
                              dtype=outputs_sorted.dtype,
                              device=outputs_sorted.device)
    boundary_low = boundaries[0].unsqueeze(0).unsqueeze(1)
    boundary_high = boundaries[1].unsqueeze(0).unsqueeze(1)
    outputs_highside = torch.cat((outputs_clamped, boundary_high), dim=0)
    outputs_lowside = torch.cat((boundary_low, outputs_clamped), dim=0)

    # Most intervals are multiplied by 0.5 because they are shared between two
    # neighbours
    # The first and last interval do not get divided bu two because they are
    # not shared
    multiplier = 0.5 * torch.ones_like(outputs_highside)
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
        return dist_nn[0]  # entry 0 is the tensor, entry 1 are the indices
    elif mode == "double_nn":
        return dist
    elif mode == "intervals":
        # Determine the intervals between the points, up and down together.
        intervals = dist[1:] + dist[:-1]
        return intervals
