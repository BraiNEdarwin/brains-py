# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 13:14:52 2019

@author: HCRuiz and Unai Alegre
"""
import torch
import numpy as np
from brainspy.algorithms.modules.performance.accuracy import get_accuracy
from brainspy.utils.pytorch import TorchUtils
import warnings

# TODO: implement corr_lin_fit (AF's last fitness function)?

# Description of the file: Set of functions to measure separability and similarity of signals

# %% Accuracy of a perceptron as fitness: measures separability


def accuracy_fit(output, target, default_value=False):
    if default_value:
        return 0
        # print(f'Clipped at {clipvalue} nA')
    else:
        acc, _, _ = get_accuracy(output, target)
        return acc


# %% Correlation between output and target: measures similarity


def corr_fit(output, target, default_value=False):
    if default_value:
        # print(f'Clipped at {clipvalue} nA')
        return -1
    else:
        return pearsons_correlation(output[:, 0], target[:, 0])


# %% Combination of a sigmoid with pre-defined separation threshold (2.5 nA) and
# the correlation function. The sigmoid can be adapted by changing the function 'sig( , x)'


def corrsig_fit(output, target, default_value=False):
    if default_value:
        # print(f'Clipped at {torch.abs(output)} nA')
        return -1
    else:
        corr = pearsons_correlation(output[:, 0], target[:, 0])
        sep = output[target == 1].mean() - output[target == 0].mean()
        sig = torch.sigmoid(-2 * (sep - 2))
        return corr * sig


def pearsons_correlation(x, y):
    vx = x - x.mean(dim=0)
    vy = y - y.mean(dim=0)
    return torch.sum(vx * vy) / (
        torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2))
    )


def corrsig(output, target):
    corr = torch.mean((output - torch.mean(output)) * (target - torch.mean(target))) / (
        torch.std(output) * torch.std(target) + 1e-10
    )
    x_high_min = torch.min(output[(target == 1)])
    x_low_max = torch.max(output[(target == 0)])
    delta = x_high_min - x_low_max
    return (1.1 - corr) / torch.sigmoid((delta - 5) / 3)


def sqrt_corrsig(output, target):
    corr = torch.mean((output - torch.mean(output)) * (target - torch.mean(target))) / (
        torch.std(output) * torch.std(target) + 1e-10
    )
    x_high_min = torch.min(output[(target == 1)])
    x_low_max = torch.max(output[(target == 0)])
    delta = x_high_min - x_low_max
    # 5/3 works for 0.2 V gap
    return (1.0 - corr) ** (1 / 2) / torch.sigmoid((delta - 2) / 5)


def fisher_fit(output, target, default_value=False):
    if default_value:
        return 0
    else:
        return -fisher(output, target)


def fisher(output, target):
    """Separates classes irrespective of assignments.
    Reliable, but insensitive to actual classes"""
    x_high = output[(target == 1)]
    x_low = output[(target == 0)]
    m0, m1 = torch.mean(x_low), torch.mean(x_high)
    s0, s1 = torch.var(x_low), torch.var(x_high)
    mean_separation = (m1 - m0) ** 2
    return -mean_separation / (s0 + s1)


def fisher_added_corr(output, target):
    x_high = output[(target == 1)]
    x_low = output[(target == 0)]
    m0, m1 = torch.mean(x_low), torch.mean(x_high)
    s0, s1 = torch.var(x_low), torch.var(x_high)
    mean_separation = (m1 - m0) ** 2
    corr = torch.mean((output - torch.mean(output)) * (target - torch.mean(target))) / (
        torch.std(output) * torch.std(target) + 1e-10
    )
    return (1 - corr) - 0.5 * mean_separation / (s0 + s1)


def fisher_multipled_corr(output, target):
    x_high = output[(target == 1)]
    x_low = output[(target == 0)]
    m0, m1 = torch.mean(x_low), torch.mean(x_high)
    s0, s1 = torch.var(x_low), torch.var(x_high)
    mean_separation = (m1 - m0) ** 2
    corr = torch.mean((output - torch.mean(output)) * (target - torch.mean(target))) / (
        torch.std(output) * torch.std(target) + 1e-10
    )
    return (1 - corr) * (s0 + s1) / mean_separation


def sigmoid_nn_distance(outputs, target=None):
    # Sigmoid nearest neighbour distance: a squeshed version of a sum of all internal distances between points.
    if target != None:
        raise Warning('This loss function does not use target values. Target ignored.')
    dist_nn = get_clamped_intervals(outputs, mode='single_nn')
    return -1 * torch.mean(torch.sigmoid(dist_nn / 2) - 0.5)


def get_clamped_intervals(outputs, mode, boundaries=[-352, 77]):
    # First we sort the output, and clip the output to a fixed interval.
    outputs_sorted = outputs.sort(dim=0)[0]
    outputs_clamped = outputs_sorted.clamp(boundaries[0], boundaries[1])

    # THen we prepare two tensors which we subtract from each other to calculate nearest neighbour distances.
    boundaries = torch.tensor(boundaries, dtype=outputs_sorted.dtype)
    boundary_low = boundaries[0].unsqueeze(0).unsqueeze(1)
    boundary_high = boundaries[1].unsqueeze(0).unsqueeze(1)
    outputs_highside = torch.cat((outputs_clamped, boundary_high), dim=0)
    outputs_lowside = torch.cat((boundary_low, outputs_clamped), dim=0)

    # Most intervals are multiplied by 0.5 because they are shared between two neighbours
    # The first and last interval do not get divided bu two because they are not shared
    multiplier = 0.5 * torch.ones_like(outputs_highside)
    multiplier[0] = 1
    multiplier[-1] = 1

    # Calculate the actual distance between points
    dist = (outputs_highside - outputs_lowside) * multiplier

    if mode == 'single_nn':
        # Only give nearest neighbour (single!) distance
        dist_nns = torch.cat((dist[1:], dist[:-1]), dim=1)  # both nearest neighbours
        dist_nn = torch.min(dist_nns, dim=1)  # only the closes nearest neighbour
        return dist_nn[0]  # entry 0 is the tensor, entry 1 are the indices
    elif mode == 'double_nn':
        return dist
    elif mode == 'intervals':
        # Determine the intervals between the points, up and down together.
        intervals = dist[1:] + dist[:-1]
        return intervals
