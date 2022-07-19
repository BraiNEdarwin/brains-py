"""
Set of fitness functions for genetic algorithm and loss functions for gradient
descent.
"""
import warnings

import torch

from brainspy.utils.pytorch import TorchUtils
from brainspy.utils.performance.accuracy import get_accuracy


def accuracy_fit(output: torch.Tensor,
                 target: torch.Tensor,
                 default_value=False) -> torch.Tensor:
    """
    Fitness function for genetic algorithm using accuracy of a perceptron.
    Teaches single perceptron to transform output to target and
    evaluates the accuracy; is a percentage.
    Will return default value (0) if indicated.

    Needs at least 10 datapoints in each signal.

    Example
    -------
    >>> accuracy_fit(torch.rand((100, 3)), torch.rand(100, 3))
    torch.Tensor([48., 21.2, 3.5])
    >>> accuracy_fit(torch.rand((100, 3)), torch.rand(100, 3), True)
    torch.Tensor([0.0, 0.0, 0.0])

    Parameters
    ----------
    output : torch.Tensor
        The output data, shape [n, m] with m signals of n datapoints.
    target : torch.Tensor
        The target data, shape [n, m] with m signals of n datapoints.
    default_value : bool, optional
        Return the default value or not, by default False.

    Returns
    -------
    torch.Tensor
        Default value or calculated fitness for each pair of signals.

    Raises
    ------
    AssertionError
        If dimensions of output and target are not the same.
    """
    if type(output) != torch.Tensor or type(target) != torch.Tensor or type(
            default_value) != bool:
        raise AssertionError("Invalid type for arguments provided")
    assert output.shape == target.shape, "Dimensions of data are different."

    if default_value:
        return torch.zeros(output.shape[1],
                           device=output.device,
                           dtype=output.dtype)
    else:
        result = torch.zeros(output.shape[1],
                             device=output.device,
                             dtype=output.dtype)
        for i in range(output.shape[1]):
            result[i] = get_accuracy(output[:, i].unsqueeze(1),
                                     target[:,
                                            i].unsqueeze(1))["accuracy_value"]
        return result


def corr_fit(output: torch.Tensor,
             target: torch.Tensor,
             default_value=False) -> torch.Tensor:
    """
    Fitness function for genetic algorithm using Pearson correlation.
    See pearsons_correlation for more info.
    Will return default value (-1) if indicated.

    Example
    -------
    >>> corr_fit(torch.rand((100, 3)), torch.rand(100, 3))
    torch.Tensor([0.5, 0.4, -0.34])
    >>> corr_fit(torch.rand((100, 3)), torch.rand(100, 3), True)
    torch.Tensor([-1.0, -1.0, -1.0])

    Parameters
    ----------
    output : torch.Tensor
        The output data, shape [n, m] with m signals of n datapoints.
    target : torch.Tensor
        The target data, shape [n, m] with m signals of n datapoints.
    default_value : bool, optional
        Return the default value or not, by default False.

    Returns
    -------
    torch.Tensor
        Default value or calculated fitness for each pair of signals.

    Raises
    ------
    AssertionError
        If dimensions of output and target are not the same.
    """
    if type(output) != torch.Tensor or type(target) != torch.Tensor or type(
            default_value) != bool:
        raise AssertionError("Invalid type for arguments provided")
    assert output.shape == target.shape, "Dimensions of data are different."
    if default_value:
        return -torch.ones(
            output.shape[1], device=output.device, dtype=output.dtype)
    else:
        return pearsons_correlation(output, target)


def corrsig_fit(output: torch.Tensor,
                target: torch.Tensor,
                default_value=False,
                sigmoid_center=0,
                sigmoid_scale=1) -> torch.Tensor:
    """
    Fitness function for genetic algorithm using correlation and a sigmoid
    function.
    Will return default value (-1) if indicated.

    For values of parameters see this paper:
    https://www.nature.com/articles/s41565-020-00779-y

    Note: target data must be binary for this to work.

    Example
    -------
    >>> corrsig_fit(torch.rand((100, 3)), torch.round(torch.rand(100, 3)))
    torch.Tensor([0.5, 0.4, -0.34])
    >>> corrsig_fit(torch.rand((100, 3)), torch.round(torch.rand(100, 3)),
                    True)
    torch.Tensor([-1.0, -1.0, -1.0])

    Parameters
    ----------
    output : torch.Tensor
        The output data, shape [n, m] with m signals of n datapoints.
    target : torch.Tensor
        The target data, shape [n, m] with m signals of n datapoints;
        should be binary.
    default_value : bool, optional
        Return the default value or not, by default False.
    sigmoid_center : float
        Shift of the sigmoid, by default 0.
    sigmoid_scale : float
        Scale of the sigmoid, by default 1.

    Returns
    -------
    torch.Tensor
        Default value or calculated fitness for each pair of signals.
        Will be NaN if target data is not binary.

    Raises
    ------
    AssertionError
        If dimensions of output and target are not the same.
    """
    if type(output) != torch.Tensor or type(target) != torch.Tensor or type(
            default_value) != bool:
        raise AssertionError("Invalid type for arguments provided")
    if default_value:
        return -torch.ones(
            output.shape[1], device=output.device, dtype=output.dtype)
    else:
        assert output.shape == target.shape, "Dimensions of data are different."
        corr = pearsons_correlation(output, target)
        sig = torch.zeros(output.shape[1],
                          device=output.device,
                          dtype=output.dtype)
        for i in range(output.shape[1]):
            sep = output[:, i][target[:, i] == 1].mean() - output[:, i][
                target[:, i] == 0].mean()
            sig[i] = torch.sigmoid(sigmoid_scale * (sep - sigmoid_center))
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
        Dataset, shape [n, m] with m signals of n datapoints.
    y : torch.Tensor
        Dataset, shape [n, m] with m signals of n datapoints.

    Returns
    -------
    torch.Tensor
        Correlation between x and y for each pair of signals.
        Will be nan if a data is uniform.

    Raises
    ------
    AssertionError
        If dimensions of x and y are not the same.
    UserWarning
        If result is nan (which happens if a dataset has variance 0, is
        uniform).
    """
    if type(x) != torch.Tensor or type(y) != torch.Tensor:
        raise AssertionError("Invalid type for arguments provided")
    assert x.shape == y.shape, "Dimensions of data are different."
    vx = x - x.mean(dim=0)
    vy = y - y.mean(dim=0)
    sum_vx = torch.sum(vx**2, dim=0)
    sum_vy = torch.sum(vy**2, dim=0)
    sum_vxy = torch.sum(vx * vy, dim=0)
    if 0.0 in sum_vx or 0.0 in sum_vy:
        warnings.warn("Variance of dataset is 0, correlation is nan.")
    return sum_vxy / (torch.sqrt(sum_vx) * torch.sqrt(sum_vy))


def corrsig(output: torch.Tensor,
            target: torch.Tensor,
            sigmoid_center: float = 0,
            sigmoid_scale: float = 1,
            corr_shift: float = 1.1) -> torch.Tensor:
    """
    Loss function for gradient descent using a sigmoid function.

    For values of parameters see this paper:
    https://www.nature.com/articles/s41565-020-00779-y

    Example
    -------
    >>> corrsig(torch.rand((100, 1)), torch.round(torch.rand((100, 1))))
    torch.Tensor(2.5)

    Parameters
    ----------
    output : torch.Tensor
        The output data, shape [n, m] with m signals of n datapoints.
    target : torch.Tensor
        The target data, shape [n, m] with m signals of n datapoints;
        should be binary.
    sigmoid_center : float
        Center of the sigmoid.
    sigmoid_scale : float
        Scale of the sigmoid, between 0 and 1.
    corr_shift : float
        Shifting the correlation value.

    Returns
    -------
    torch.Tensor
        Value of loss function for each pair of signals.

    Raises
    ------
    AssertionError
        If dimensions of x and y are not the same.
    """
    if type(output) != torch.Tensor or type(target) != torch.Tensor:
        raise AssertionError("Invalid type for arguments provided")
    assert output.shape == target.shape, "Dimensions of data are different."
    corr = pearsons_correlation(output, target)
    # difference between smallest false negative and largest false positive
    delta = torch.zeros(output.shape[1],
                        device=output.device,
                        dtype=output.dtype)
    for i in range(output.shape[1]):
        x_high_min = torch.min(output[:, i][target[:, i] == 1])
        x_low_max = torch.max(output[:, i][(target[:, i] == 0)])
        delta[i] = x_high_min - x_low_max

    return (corr_shift - corr) / torch.sigmoid(
        (delta - sigmoid_center) / sigmoid_scale)


def fisher_fit(output: torch.Tensor,
               target: torch.Tensor,
               default_value=False) -> torch.Tensor:
    """
    Fitness function for genetic algorithm using the negative of the
    Fisher linear discriminant. For more information see fisher method.

    Can return default value (0).

    Example
    -------
    >>> fisher_fit(torch.rand((100, 3)), torch.rand((100, 3)),
                   False)
    torch.Tensor([2.5, 1.2, 0.5])
    >>> fisher_fit(torch.rand((100, 3)), torch.rand((100, 3)),
                   True)
    torch.Tensor([0.0, 0.0, 0.0])

    Parameters
    ----------
    output : torch.Tensor
        The output data, shape [n, m] with m signals of n datapoints.
    target : torch.Tensor
        The target data, shape [n, m] with m signals of n datapoints;
        should be binary.
    default_value : bool, optional
        Return the default value or not, by default False.

    Returns
    -------
    torch.Tensor
        Default value or calculated fitness for each pair of signals.

    Raises
    ------
    AssertionError
        If dimensions of x and y are not the same.
    """
    if type(output) != torch.Tensor or type(target) != torch.Tensor or type(
            default_value) != bool:
        raise AssertionError("Invalid type for arguments provided")
    assert output.shape == target.shape, "Dimensions of data are different."
    if default_value:
        return torch.zeros(output.shape[1],
                           device=output.device,
                           dtype=output.dtype)
    else:
        return fisher(output, target)


def fisher(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Calculate the negative of the Fisher linear discriminant between
    two datasets. Used as a loss function for gradient descent.

    More information here:
    https://sthalles.github.io/fisher-linear-discriminant/

    Example
    -------
    >>> fisher(torch.rand((100, 3)), torch.rand((100, 3)),
                   False)
    torch.Tensor([2.5, 1.2, 0.5])

    Parameters
    ----------
    output : torch.Tensor
        Dataset, shape [n, m] with m signals of n datapoints.
    target : torch.Tensor
        Dataset, shape [n, m] with m signals of n datapoints;
        should be binary.

    Returns
    -------
    torch.Tensor
        Value of Fisher linear discriminant for each pair of signals.

    Raises
    ------
    AssertionError
        If dimensions of x and y are not the same.
    UserWarning
        If result is nan (which happens if a dataset has variance 0, is
        uniform).
    """
    if type(output) != torch.Tensor or type(target) != torch.Tensor:
        raise AssertionError("Invalid type for arguments provided")
    assert output.shape == target.shape, "Dimensions of data are different."
    result = torch.zeros(output.shape[1],
                         device=output.device,
                         dtype=output.dtype)
    for i in range(output.shape[1]):
        x_high = output[:, i][(target[:, i] == 1)]
        x_low = output[:, i][(target[:, i] == 0)]
        m0, m1 = torch.mean(x_low), torch.mean(x_high)
        s0, s1 = torch.var(x_low), torch.var(x_high)
        if 0.0 in s0 or 0.0 in s1:
            warnings.warn("Variance of dataset is 0, correlation is nan.")
        mean_separation = (m1 - m0)**2
        result[i] = mean_separation / (s0 + s1)
    return -result


def sigmoid_nn_distance(output: torch.Tensor,
                        target: torch.Tensor = None,
                        sigmoid_center: float = 0.5,
                        sigmoid_scale: float = 2.0) -> torch.Tensor:
    """
    Sigmoid of nearest neighbour distance: a squashed version of a sum of all
    internal distances between points.
    Used as a loss function for gradient descent.

    For values of parameters see this paper:
    https://www.nature.com/articles/s41565-020-00779-y

    Example
    -------
    >>> sigmoid_nn_distance(torch.rand((100, 3)))
    torch.Tensor([20.0, 11.0, 10.0])

    Parameters
    ----------
    output : torch.Tensor
        The output data, shape [n, m] with m signals of n datapoints.
    target : torch.Tensor
        The target data, will not be used.
    sigmoid_center : float
        Center of the sigmoid.
    sigmoid_scale : float
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
    if type(output) != torch.Tensor or type(target) != torch.Tensor:
        raise AssertionError("Invalid type for arguments provided")
    if target is not None:
        warnings.warn(
            "This loss function does not use target values. Target ignored.")
    dist_nn = get_clamped_intervals(output, mode="single_nn")
    return -1 * torch.mean(
        torch.sigmoid(dist_nn / sigmoid_scale) - sigmoid_center, dim=0)


def get_clamped_intervals(output: torch.Tensor,
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
    output : torch.Tensor
        Dataset, shape [n, m] with m signals of n datapoints.
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
    if type(output) != torch.Tensor or type(mode) != str:
        raise AssertionError("Invalid type for arguments provided")
    # First we sort the output, and clip the output to a fixed interval.
    output_sorted = output.sort(dim=0)[0]
    output_clamped = output_sorted.clamp(boundaries[0], boundaries[1])

    # Then we prepare two tensors which we subtract from each other to
    # calculate nearest neighbour distances.
    boundaries = TorchUtils.format(boundaries,
                                   device=output.device,
                                   data_type=output.dtype)
    boundary_low = boundaries[0] * torch.ones(
        [1, output.shape[1]], device=output.device, dtype=output.dtype)
    boundary_high = boundaries[1] * torch.ones(
        [1, output.shape[1]], device=output.device, dtype=output.dtype)
    output_highside = torch.cat((output_clamped, boundary_high), dim=0)
    output_lowside = torch.cat((boundary_low, output_clamped), dim=0)

    multiplier = torch.ones_like(output_highside, device=output.device)
    multiplier.type_as(output)
    multiplier[0] = 1
    multiplier[-1] = 1

    # Calculate the actual distance between points
    dist = (output_highside - output_lowside) * multiplier

    if mode == "single_nn":
        # Only give nearest neighbour (single!) distance
        return torch.minimum(dist[1:], dist[:-1])
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
