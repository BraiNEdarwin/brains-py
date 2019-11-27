#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 14:36:36 2018
Generate a piecewise linear wave form with general amplitudes and intervals.
@author: hruiz and ualegre
"""
import numpy as np
import warnings


def safety_format(amplitudes_input, lengths_input, slopes_input):
    '''The aim of this function is to start and end the waveform signal with zero,
     in order to avoid damaging the boron-doped silicon device.'''
    amplitudes_output = list(amplitudes_input)  # Clone the list
    amplitudes_output.append(0)

    if type(slopes_input) is int:
        slopes_output = [slopes_input] * len(amplitudes_output)

    if type(lengths_input) is int:
        lengths_output = [lengths_input] * (len(amplitudes_output) - 1)
    else:
        lengths_output = list(lengths_input)

    lengths_output.append(0)

    output = np.linspace(0, amplitudes_output[0], slopes_output[0])  # np.concatenate((np.array([]), np.linspace(0, amplitudes_output[0], slopes_output[0])))
    return output, amplitudes_output, lengths_output, slopes_output


def generate_waveform(amplitudes, amplitude_lengths, slope_lengths=0, safety_formatting=True):
    '''
    Generates a waveform (voltage input over time) with constant intervals of value amplitudes[i] for interval i of length[i].

    amplitudes = The input from which a waveform will be generated. The input is in form of a list.
    amplitude_lengths = The number of points used to represent the amplitudes. It can be provided as a single number or as a list in which all the length values will correspond to its corresponding amplitude value.
    slope_lengths = The number of points of the slope.

    The output is in list format
    '''
    if safety_formatting is True:
        output, amplitudes, amplitude_lengths, slope_lengths = safety_format(amplitudes, amplitude_lengths, slope_lengths)
    else:
        output = np.array([])
        warnings.warn('WARNING: Safety formatting is not enabled. This can make the boron-doped silicon device unusable. ')
    if len(amplitudes) == len(amplitude_lengths) == len(slope_lengths):
        for i in range(len(amplitudes) - 1):
            output = np.concatenate((output, np.array([amplitudes[i]] * amplitude_lengths[i])))
            output = np.concatenate((output, np.linspace(amplitudes[i], amplitudes[i + 1], slope_lengths[i])))
    else:
        assert False, 'Assignment of amplitudes and lengths/slopes is not unique!'

    return output


def generate_mask(amplitudes, amplitude_lengths, slope_lengths=0, safety_formatting=True):
    '''
    Generates a waveform (voltage input over time) with constant intervals of value amplitudes[i] for interval i of length[i].

    amplitudes = The input from which a waveform will be generated. The input is in form of a list.
    amplitude_lengths = The number of points used to represent the amplitudes. It can be provided as a single number or as a list in which all the length values will correspond to its corresponding amplitude value.
    slope_lengths = The number of points of the slope.

    The output is in list format
    '''
    mask = []
    if safety_formatting is True:
        _, amplitudes, amplitude_lengths, slope_lengths = safety_format(amplitudes, amplitude_lengths, slope_lengths)
    else:
        warnings.warn('WARNING: Safety formatting is not enabled. This can make the boron-doped silicon device unusable. ')
    if len(amplitudes) == len(amplitude_lengths) == len(slope_lengths):
        mask += [False] * slope_lengths[0]
        for i in range(len(amplitudes) - 1):
            mask += [True] * amplitude_lengths[i]
            mask += [False] * slope_lengths[i]
    else:
        assert False, 'Assignment of amplitudes and lengths/slopes is not unique!'

    return mask


def generate_slopped_plato(slope_length, total_length, value=1):
    length = total_length - (2 * slope_length)
    up = np.linspace(0, value, slope_length)
    down = np.linspace(value, 0, slope_length)
    plato = np.broadcast_to(value, length)
    return np.concatenate((up, plato, down))


if __name__ == '__main__':

    from matplotlib import pyplot as plt
    amplitudes, lengths = [3, 1, -1, 1], 100
    wave = generate_waveform(amplitudes, lengths, slope_lengths=30)
    print(len(wave))
    plt.figure()
    plt.plot(wave)
    plt.show()
