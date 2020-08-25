#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate a piecewise linear wave form with general amplitudes and intervals.
"""

import torch
import numpy as np


class WaveformManager():
    def __init__(self, configs):
        self.amplitude_lengths = configs['amplitude_lengths']
        self.slope_lengths = configs['slope_lengths']

    def _expand(self, parameter, length):
        '''The aim of this function is to format the amplitudes and slopes to have the same length as the amplitudes,
        in case they are specified with an integer number.'''
        # output_data = list(data)
        if type(parameter) is int:
            return [parameter] * length
        return parameter

    def points_to_waveform(self, data):
        '''
        Generates a waveform (voltage input over time) with constant intervals of value amplitudes[i] for interval i of length[i].

        amplitudes = The input from which a waveform will be generated. The input is in form of a list.
        amplitude_lengths = The number of points used to represent the amplitudes. It can be provided as a single number or as a list in which all the length values will correspond to its corresponding amplitude value.
        slope_lengths = The number of points of the slope.

        The output is in list format
        '''
        output = np.ndarray([])
        # data = list(self.safety_format(data, safety_formatting))
        amplitude_lengths = self._expand(self.amplitude_lengths, len(data))
        slope_lengths = self._expand(self.slope_lengths, len(data))
        # amplitudes, amplitude_lengths, slope_lengths = self.format_amplitudes_and_slopes(amplitudes, self.amplitude_lengths, self.slope_lengths)

        if len(data) == len(amplitude_lengths) == len(slope_lengths):
            output = np.linspace(0, data[0], slope_lengths[0])
            data_size = len(data) - 1
            for i in range(data_size):
                output = np.concatenate((output, np.array(([data[i]] * amplitude_lengths[i]))))
                output = np.concatenate((output, np.linspace(data[i], data[i + 1], slope_lengths[i])))
            i = data_size
            output = np.concatenate((output, np.array(([data[i]] * amplitude_lengths[i]))))
            output = np.concatenate((output, np.linspace(data[i], 0, slope_lengths[i])))

        else:
            assert False, 'Assignment of amplitudes and lengths/slopes is not unique!'
        return output

    # def points_to_plateau(self, data):
    #     # output = np.ndarray([])
    #     amplitude_lengths = self._expand(self.amplitude_lengths, len(data))
    #     output = np.array(([data[0]] * amplitude_lengths[0]))
    #     for i in range(1, len(data)):
    #         output = np.concatenate((output, np.array(([data[i]] * amplitude_lengths[i]))))
    #     return output

    def points_to_plateaus(self, data):
        # output = np.ndarray([])
        result = data[0].repeat(self.amplitude_lengths, 1)
        for i in range(1, len(data)):
            result = torch.cat((result, data[i].repeat(self.amplitude_lengths, 1)), dim=0)
        # amplitude_lengths = self._expand(self.amplitude_lengths, len(data))
        # output = data[0].expand(data.shape[0] * amplitude_lengths[0], -1)
        # for i in range(1, data.shape[1]):
        #     output = torch.cat((output, data[i].expand(data.shape[0] * amplitude_lengths[i], -1)))
        return result

    def plateaus_to_waveform(self, data):
        '''
        Generates a waveform (voltage input over time) with constant intervals of value amplitudes[i] for interval i of length[i].

        amplitudes = The input from which a waveform will be generated. The input is in form of a list.
        amplitude_lengths = The number of points used to represent the amplitudes. It can be provided as a single number or as a list in which all the length values will correspond to its corresponding amplitude value.
        slope_lengths = The number of points of the slope.

        The output is in list format
        '''
        output = np.ndarray([])
        point_length = int(len(data) / self.amplitude_lengths)
        amplitude_lengths = self._expand(self.amplitude_lengths, point_length)
        slope_lengths = self._expand(self.slope_lengths, point_length)
        i = 0
        j = 0
        mask = []
        output = np.linspace(0, data[0], slope_lengths[0])
        mask += [False] * slope_lengths[0]
        data_size = point_length - 1
        for i in range(data_size):
            current_plateau = np.array(data[j:j + amplitude_lengths[i]])
            mask += [True] * len(current_plateau)
            current_slope = np.linspace(data[j + amplitude_lengths[i] - 1], data[j + amplitude_lengths[i]], slope_lengths[i])
            mask += [False] * len(current_slope)
            output = np.concatenate((output, current_plateau))
            output = np.concatenate((output, current_slope))
            j += amplitude_lengths[i]
        i = data_size
        current_plateau = np.array(data[j:j + amplitude_lengths[i]])
        current_slope = np.linspace(data[j + amplitude_lengths[i] - 1], 0, slope_lengths[i])
        mask += [True] * len(current_plateau)
        mask += [False] * len(current_slope)
        output = np.concatenate((output, current_plateau))
        output = np.concatenate((output, current_slope))
        return output, mask

    def plateaus_to_points(self, data):
        amplitude_lengths = self._expand(self.amplitude_lengths, int(len(data) / self.amplitude_lengths))
        output = np.array([])
        i = 0
        for amplitude_length in amplitude_lengths:
            output = np.append(output, data[i:i + amplitude_length].mean())
            i += amplitude_length
        return output

    def waveform_to_points(self, data):
        data = data[self.generate_mask(len(data))]
        return self.plateaus_to_points(data)

    def waveform_to_plateaus(self, data):
        return data[self.generate_mask(len(data))]

    def generate_mask(self, data_size):
        '''
        Generates a waveform (voltage input over time) with constant intervals of value amplitudes[i] for interval i of length[i].

        amplitudes = The input from which a waveform will be generated. The input is in form of a list.
        amplitude_lengths = The number of points used to represent the amplitudes. It can be provided as a single number or as a list in which all the length values will correspond to its corresponding amplitude value.
        slope_lengths = The number of points of the slope.

        The output is in list format
        '''
        assert isinstance(self.slope_lengths, int) and isinstance(self.amplitude_lengths, int), "Generate mask operation is only supported by integer slope and amplitude lengths"
        mask = []
        # amplitude_lengths = self.expand(self.amplitude_lengths, data_length)
        # slope_lengths = self.expand(self.slope_lengths, data_length)
        # if data_length == len(amplitude_lengths) == len(slope_lengths):
        i = 0
        odd = True
        while i < data_size:
            if odd:
                mask += [False] * self.slope_lengths
                i += self.slope_lengths
                odd = False
            else:
                mask += [True] * self.amplitude_lengths
                i += self.amplitude_lengths
                odd = True
        return mask

    def generate_slopped_plateau(self, total_length, value=1):
        length = total_length - (2 * self.slope_lengths)
        ramping_up = np.linspace(0, value, self.slope_lengths)
        ramping_down = np.linspace(value, 0, self.slope_lengths)
        plato = np.broadcast_to(value, length)
        return np.concatenate((ramping_up, plato, ramping_down))
