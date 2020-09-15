""" This module is part of the utils of brains-py helps managing
    the waveforms of the signals sent to and
    received by the hardware DNPUs (Dopant Network Processing Units).
"""
import torch
from brainspy.utils.pytorch import TorchUtils
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
        self.plateau_length = configs["plateau_length"]
        self.slope_length = configs["slope_length"]
        self.generate_mask_base()

    def generate_mask_base(self):
        mask = []
        final_mask = [False] * self.slope_length
        mask += final_mask
        mask += [True] * self.plateau_length
        self.initial_mask = torch.tensor(mask)
        self.final_mask = torch.tensor(final_mask)

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

        plateaus = The input from which a waveform will be generated. The input is in form of a list.
        plateau_length = The number of points used to represent the amplitudes. It can be provided as a single
        number or as a list in which all the length values will correspond to its corresponding amplitude value.
        slope_lengths = The number of points of the slope.

        The output is in list format
        """
        data_size = len(data) - 1
        # output = torch.tensor([])
        # data = list(self.safety_format(data, safety_formatting))
        # plateau_lengths = self._expand(self.plateau_lengths, len(data))
        # slope_lengths = self._expand(self.slope_lengths, len(data))
        # amplitudes, plateau_lengths, slope_lengths = self.format_amplitudes_and_slopes(amplitudes, self.plateau_lengths, self.slope_lengths)
        tmp = TorchUtils.get_numpy_from_tensor(data)
        # if len(data) == len(plateau_lengths) == len(slope_lengths):
        output = TorchUtils.get_tensor_from_numpy(np.linspace(0, tmp[0], self.slope_length))
        for i in range(data_size):
            output = torch.cat((output, data[i].repeat(self.plateau_length, 1)))
            output = torch.cat(
                (output, TorchUtils.get_tensor_from_numpy(np.linspace(tmp[i], tmp[i + 1], self.slope_length)))
            )
        output = torch.cat((output, data[-1].repeat(self.plateau_length, 1)))
        output = torch.cat((output, TorchUtils.get_tensor_from_numpy(np.linspace(tmp[-1], 0, self.slope_length))))
        del tmp
        # else:
        #     assert False, "Assignment of amplitudes and lengths/slopes is not unique!"
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
        # result = data[0].repeat(self.plateau_length, 1)
        # for i in range(1, len(data)):
        #     result = torch.cat((result, data[i].repeat(self.plateau_length, 1)), dim=0)
        # plateau_lengths = self._expand(self.plateau_lengths, len(data))
        # output = data[0].expand(data.shape[0] * plateau_lengths[0], -1)
        # for i in range(1, data.shape[1]):
        #     output = torch.cat((output, data[i].expand(data.shape[0] * plateau_lengths[i], -1)))
        return self.tile(data, 0, self.plateau_length)

    def tile(self, t, dim, n_tile):
        init_dim = t.size(dim)
        repeat_idx = [1] * t.dim()
        repeat_idx[dim] = n_tile
        t = t.repeat(*(repeat_idx))
        order_index = torch.cat([init_dim * torch.arange(n_tile, device=t.device, dtype=torch.long) + i for i in range(init_dim)])
        return torch.index_select(t, dim, order_index)

    def plateaus_to_waveform(self, data, return_pytorch=True):
        """
        Generates a waveform (voltage input over time) with constant intervals of value amplitudes[i] for interval i of length[i].

        amplitudes = The input from which a waveform will be generated. The input is in form of a list.
        plateau_lengths = The number of points used to represent the amplitudes. It can be provided as a single number or as a list in which all the length values will correspond to its corresponding amplitude value.
        slope_lengths = The number of points of the slope.

        The output is in list format
        """
        # The function np.linspace supports multiple dimension while torch.linspace does not. Data is transformed from tensor to numpy and then returned back to tensor.
        assert (len(data) % self.plateau_length == 0), f'Incorrect data shape with respect to plateau length {self.plateau_length}.'
        data_size = int(len(data) / self.plateau_length) - 1
        tmp = TorchUtils.get_numpy_from_tensor(data)
        output = np.ndarray([])
        # data = list(self.safety_format(data, safety_formatting))
        # plateau_lengths = self._expand(self.plateau_length, len(tmp))
        # slope_lengths = self._expand(self.slope_length, len(tmp))
        # amplitudes, plateau_lengths, slope_lengths = self.format_amplitudes_and_slopes(amplitudes, self.plateau_lengths, self.slope_lengths)

        # if len(tmp) == len(self.plateau_length) == len(self.slope_length):
        start = 0
        output = np.linspace(0, tmp[start], self.slope_length)
        mask = []
        mask += [False] * self.slope_length
        for i in range(data_size):
            end = start + self.plateau_length
            mask += [True] * self.plateau_length
            output = np.concatenate(
                (output, tmp[start:end])
            )
            mask += [False] * self.slope_length
            output = np.concatenate(
                (output, np.linspace(tmp[end - 1], tmp[end], self.slope_length))
            )
            start = end
        mask += [True] * self.plateau_length
        output = np.concatenate(
            (output, tmp[start:])
        )
        mask += [False] * self.slope_length
        output = np.concatenate((output, np.linspace(tmp[-1], 0, self.slope_length)))

        # else:
        #     assert False, "Assignment of amplitudes and lengths/slopes is not unique!"
        if return_pytorch:
            return TorchUtils.get_tensor_from_numpy(output), TorchUtils.get_tensor_from_list(mask)
        else:
            return output, mask

    def plateaus_to_points(self, data):
        # plateau_lengths = self._expand(
        #     self.plateau_lengths, int(len(data) / self.plateau_lengths)
        # )
        # output = np.array([])
        # j = 0
        # for i in range(len(data)):
        #     output = np.append(output, data[j: j + self.plateau_length].mean())
        #     j += self.plateau_length
        assert (len(data) % self.plateau_length == 0), f'Incorrect data shape {data.shape} with respect to the number of points for plateau length {self.plateau_length}.'
        point_no = int(len(data) / self.plateau_length)
        result = data.view(point_no, self.plateau_length, data.shape[1]).mean(dim=1)
        if len(result.shape) == 1:
            result = result.unsqueeze(dim=1)
        return result

    def waveform_to_points(self, data, mask=None):
        if mask is None:
            mask = self.generate_mask(len(data))
        return self.plateaus_to_points(data[mask])

    def waveform_to_plateaus(self, data, mask=None):
        if mask is None:
            mask = self.generate_mask(len(data))
        return data[mask]

    def generate_mask(self, data_size):
        """
        Generates a waveform (voltage input over time) with constant intervals of value amplitudes[i] for interval i of length[i].

        amplitudes = The input from which a waveform will be generated. The input is in form of a list.
        plateau_lengths = The number of points used to represent the amplitudes. It can be provided as a single number or as a list in which all the length values will correspond to its corresponding amplitude value.
        slope_lengths = The number of points of the slope.

        The output is in list format
        """
        # assert isinstance(self.slope_length, int) and isinstance(
        #     self.plateau_length, int
        # ), "Generate mask operation is only supported by integer slope and amplitude lengths"
        # mask = []
        # plateau_lengths = self.expand(self.plateau_lengths, data_length)
        # slope_lengths = self.expand(self.slope_lengths, data_length)
        # if data_length == len(plateau_lengths) == len(slope_lengths):
        # i = 0
        # odd = True
        # mask = []
        # mask += [False] * self.slope_length
        # mask += [True] * self.slope_length
        # repetitions = ((data_size - self.slope_length) / (self.slope_length + self.plateau_length) - 1)
        # torch.tensor(mask).repeat(repetitions)
        # while i < data_size:
        #     if odd:
        #         mask += [False] * self.slope_length
        #         i += self.slope_length
        #         odd = False
        #     else:
        #         mask += [True] * self.plateau_length
        #         i += self.plateau_length
        #         odd = True
        # return mask
        repetitions = int(
            (
                (data_size - self.slope_length)
                / (self.slope_length + self.plateau_length)
            )
        )  # -1 ?
        mask = torch.tensor(self.initial_mask).repeat(repetitions)
        return torch.cat((mask, self.final_mask))

    # def generate_slopped_plateau(self, total_length, value=1):
    #     length = total_length - (2 * self.slope_length)
    #     ramping_up = np.linspace(0, value, self.slope_length)
    #     ramping_down = np.linspace(value, 0, self.slope_length)
    #     plato = np.broadcast_to(value, length)
    #     return np.concatenate((ramping_up, plato, ramping_down))
