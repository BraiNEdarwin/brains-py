""" This module is part of the utils of brains-py helps managing
    the waveforms of the signals sent to and
    received by the hardware DNPUs (Dopant Network Processing Units).
"""
import torch
import numpy as np


class WaveformManager:
    """This class helps managing the waveforms of the signals sent to and
    received by the hardware DNPUs (Dopant Network Processing Units).

    The waveform represents a set of points. Each of the points is represented
    with a slope, a plateau and another slope. The first slope is a line that
    goes from the previous point to the current point value. The plateau repeats
    the same point a specified number of times. The second slope is a line that
    goes from the current point to the next point. The starting and ending points
    are considered zero.

    - **parameters**, **types**, **return** and **return types**::

          :param plateau_length: The lengh of the plateaus of the waveform.
          :param slope_length: The length of the slopes of the waveform.
          :type arg1: int
          :type arg1: int

    -  The class supports the following transformations:
            * From points to plateau/waveform
            * From plateau to points/waveform
            * From waveform to plateau/points

    """

    def __init__(self, configs):
        self.plateau_lengths = configs["plateau_lengths"]
        self.slope_lengths = configs["slope_lengths"]

    def _expand(self, parameter, length):
        """The aim of this function is to format the amplitudes and
        slopes to have the same length as the amplitudes, in case
        they are specified with an integer number."""
        # output_data = list(data)
        if isinstance(parameter, int):
            return [parameter] * length
        return parameter

    def points_to_waveform(self, data):
        """
        Generates a waveform (voltage input over time) with constant intervals of value amplitudes[i] for interval i of length[i].

        amplitudes = The input from which a waveform will be generated. The input is in form of a list.
        plateau_lengths = The number of points used to represent the amplitudes. It can be provided as a single
        number or as a list in which all the length values will correspond to its corresponding amplitude value.
        slope_lengths = The number of points of the slope.

        The output is in list format
        """
        output = np.ndarray([])
        # data = list(self.safety_format(data, safety_formatting))
        plateau_lengths = self._expand(self.plateau_lengths, len(data))
        slope_lengths = self._expand(self.slope_lengths, len(data))
        # amplitudes, plateau_lengths, slope_lengths = self.format_amplitudes_and_slopes(amplitudes, self.plateau_lengths, self.slope_lengths)

        if len(data) == len(plateau_lengths) == len(slope_lengths):
            output = np.linspace(0, data[0], slope_lengths[0])
            data_size = len(data) - 1
            for i in range(data_size):
                output = np.concatenate(
                    (output, np.array(([data[i]] * plateau_lengths[i])))
                )
                output = np.concatenate(
                    (output, np.linspace(data[i], data[i + 1], slope_lengths[i]))
                )
            i = data_size
            output = np.concatenate(
                (output, np.array(([data[i]] * plateau_lengths[i])))
            )
            output = np.concatenate((output, np.linspace(data[i], 0, slope_lengths[i])))

        else:
            assert False, "Assignment of amplitudes and lengths/slopes is not unique!"
        return output

    # def points_to_plateau(self, data):
    #     # output = np.ndarray([])
    #     plateau_lengths = self._expand(self.plateau_lengths, len(data))
    #     output = np.array(([data[0]] * plateau_lengths[0]))
    #     for i in range(1, len(data)):
    #         output = np.concatenate((output, np.array(([data[i]] * plateau_lengths[i]))))
    #     return output

    def points_to_plateaus(self, data):
        # output = np.ndarray([])
        result = data[0].repeat(self.plateau_lengths, 1)
        for i in range(1, len(data)):
            result = torch.cat((result, data[i].repeat(self.plateau_lengths, 1)), dim=0)
        # plateau_lengths = self._expand(self.plateau_lengths, len(data))
        # output = data[0].expand(data.shape[0] * plateau_lengths[0], -1)
        # for i in range(1, data.shape[1]):
        #     output = torch.cat((output, data[i].expand(data.shape[0] * plateau_lengths[i], -1)))
        return result

    def plateaus_to_waveform(self, data):
        """
        Generates a waveform (voltage input over time) with constant intervals of value amplitudes[i] for interval i of length[i].

        amplitudes = The input from which a waveform will be generated. The input is in form of a list.
        plateau_lengths = The number of points used to represent the amplitudes. It can be provided as a single number or as a list in which all the length values will correspond to its corresponding amplitude value.
        slope_lengths = The number of points of the slope.

        The output is in list format
        """
        output = np.ndarray([])
        point_length = int(len(data) / self.plateau_lengths)
        plateau_lengths = self._expand(self.plateau_lengths, point_length)
        slope_lengths = self._expand(self.slope_lengths, point_length)
        i = 0
        j = 0
        mask = []
        output = np.linspace(0, data[0], slope_lengths[0])
        mask += [False] * slope_lengths[0]
        data_size = point_length - 1
        for i in range(data_size):
            current_plateau = np.array(data[j: j + plateau_lengths[i]])
            mask += [True] * len(current_plateau)
            current_slope = np.linspace(
                data[j + plateau_lengths[i] - 1],
                data[j + plateau_lengths[i]],
                slope_lengths[i],
            )
            mask += [False] * len(current_slope)
            output = np.concatenate((output, current_plateau))
            output = np.concatenate((output, current_slope))
            j += plateau_lengths[i]
        i = data_size
        current_plateau = np.array(data[j: j + plateau_lengths[i]])
        current_slope = np.linspace(
            data[j + plateau_lengths[i] - 1], 0, slope_lengths[i]
        )
        mask += [True] * len(current_plateau)
        mask += [False] * len(current_slope)
        output = np.concatenate((output, current_plateau))
        output = np.concatenate((output, current_slope))
        return output, mask

    def plateaus_to_points(self, data):
        plateau_lengths = self._expand(
            self.plateau_lengths, int(len(data) / self.plateau_lengths)
        )
        output = np.array([])
        i = 0
        for amplitude_length in plateau_lengths:
            output = np.append(output, data[i: i + amplitude_length].mean())
            i += amplitude_length
        return output

    def waveform_to_points(self, data):
        data = data[self.generate_mask(len(data))]
        return self.plateaus_to_points(data)

    def waveform_to_plateaus(self, data):
        return data[self.generate_mask(len(data))]

    def generate_mask(self, data_size):
        """
        Generates a waveform (voltage input over time) with constant intervals of value amplitudes[i] for interval i of length[i].

        amplitudes = The input from which a waveform will be generated. The input is in form of a list.
        plateau_lengths = The number of points used to represent the amplitudes. It can be provided as a single number or as a list in which all the length values will correspond to its corresponding amplitude value.
        slope_lengths = The number of points of the slope.

        The output is in list format
        """
        assert isinstance(self.slope_lengths, int) and isinstance(
            self.plateau_lengths, int
        ), "Generate mask operation is only supported by integer slope and amplitude lengths"
        mask = []
        # plateau_lengths = self.expand(self.plateau_lengths, data_length)
        # slope_lengths = self.expand(self.slope_lengths, data_length)
        # if data_length == len(plateau_lengths) == len(slope_lengths):
        i = 0
        odd = True
        while i < data_size:
            if odd:
                mask += [False] * self.slope_lengths
                i += self.slope_lengths
                odd = False
            else:
                mask += [True] * self.plateau_lengths
                i += self.plateau_lengths
                odd = True
        return mask

    def generate_slopped_plateau(self, total_length, value=1):
        length = total_length - (2 * self.slope_lengths)
        ramping_up = np.linspace(0, value, self.slope_lengths)
        ramping_down = np.linspace(value, 0, self.slope_lengths)
        plato = np.broadcast_to(value, length)
        return np.concatenate((ramping_up, plato, ramping_down))
