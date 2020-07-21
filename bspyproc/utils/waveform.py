#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate a piecewise linear wave form with general amplitudes and intervals.
"""
import numpy as np
import warnings


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

    def points_to_plateau(self, data):
        assert False, 'Function not yet implemented'
        pass

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

        output = np.linspace(0, data[0], slope_lengths[0])
        data_size = point_length - 1
        for i in range(data_size):
            output = np.concatenate((output, np.array(data[j:j + amplitude_lengths[i]])))
            output = np.concatenate((output, np.linspace(data[j + amplitude_lengths[i] - 1], data[j + amplitude_lengths[i]], slope_lengths[i])))
            j += amplitude_lengths[i]
        i = data_size
        output = np.concatenate((output, np.array(data[j:j + amplitude_lengths[i]])))
        output = np.concatenate((output, np.linspace(data[j + amplitude_lengths[i] - 1], 0, slope_lengths[i])))
        return output

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
        assert type(self.slope_lengths) is int and type(self.amplitude_lengths) is int, "Generate mask operation is only supported by integer slope and amplitude lengths"
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

    def generate_slopped_plato(self, total_length, value=1):
        length = total_length - (2 * self.slope_lengths)
        up = np.linspace(0, value, self.slope_lengths)
        down = np.linspace(value, 0, self.slope_lengths)
        plato = np.broadcast_to(value, length)
        return np.concatenate((up, plato, down))
