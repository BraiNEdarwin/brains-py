""" This module is part of the utils of brains-py helps managing
    the waveforms of the signals sent to and
    received by the hardware DNPUs (Dopant Network Processing Units).
"""
import torch
from brainspy.utils.pytorch import TorchUtils
import numpy as np
from typing import Union, Tuple


class WaveformManager:
    """
    This class helps managing the waveforms of the signals sent to and
    received by the hardware DNPUs (Dopant Network Processing Units).

    The waveform represents a set of points. Each of the points is represented
    with a slope, a plateau and another slope. The first slope is a line that
    goes from the previous point to the current point value. The plateau repeats
    the same point a specified number of times. The second slope is a line that
    goes from the current point to the next point. The starting and ending points
    are considered zero.
    """

    def __init__(self, configs):
        """
        To initialize the data from the configs dict

        Parameters
        ----------
        configs : dict
            configurations of the model

            :param plateau_length: int
                 The lengh of the plateaus of the waveform.
            :param slope_length: int
                 The length of the slopes of the waveform.

        Example
        --------
        configs = {}
        configs["plateau_length"] = 80
        configs["slope_length"] = 20
        waveform_mgr = WaveformManager(configs)

        """
    def __init__(self, configs):
        self.plateau_length = configs["plateau_length"]
        self.slope_length = configs["slope_length"]
        self.generate_mask_base()

    def generate_mask_base(self):
        """
        To generate a mask base for the torch tensor based on the slope length and plateau_length

        Example
        -------
        configs = {}
        configs["plateau_length"] = 80
        configs["slope_length"] = 20
        waveform_mgr = WaveformManager(configs)
        waveform_mgr.generate_mask_base()

        """
        mask = []
        final_mask = [False] * self.slope_length
        mask += final_mask
        mask += [True] * self.plateau_length
        self.initial_mask = torch.tensor(mask)
        self.final_mask = torch.tensor(final_mask)

    def _expand(self, parameter, length):
        """
        The aim of this function is to format the amplitudes and
        slopes to have the same length as the amplitudes, in case
        they are specified with an integer number.

        Parameters
        ----------
        parameter : int/list
            value that specifies the amplitude which can be in the form of an integer or a list
        length : int
            length of amplitude

        Returns
        -------
        list
            formatted amplitudes and slope to have same length

        Example
        -------
        parameter = 20
        length = 4
        waveform_mgr = WaveformManager(configs)
        new_parameter = waveform_mgr._expand()


        """
        if isinstance(parameter, int):
            return [parameter] * length
        return parameter

    def points_to_waveform(self, data):
        """
        Generates a waveform (voltage input over time) with constant intervals of value amplitudes[i] for interval i of length[i].

        Parameters
        ----------
        data : torch.tensor
            points for which waveform is generated as a torch tensor

        Returns
        -------
        torch.tensor
            the generated waveworm torch tesnor

        Example
        --------
        waveform_mgr = WaveformManager(configs)
        data = (1,1)
        points = torch.rand(data)
        waveform = waveform_mgr.points_to_waveform(points)

        """
        data_size = len(data) - 1
        tmp = TorchUtils.to_numpy(data)
        output = TorchUtils.format(np.linspace(0, tmp[0], self.slope_length))
        for i in range(data_size):
            output = torch.cat((output, data[i].repeat(self.plateau_length, 1)))
            output = torch.cat(
                (
                    output,
                    TorchUtils.format(
                        np.linspace(tmp[i], tmp[i + 1], self.slope_length)
                    ),
                )
            )
        output = torch.cat((output, data[-1].repeat(self.plateau_length, 1)))
        output = torch.cat(
            (output, TorchUtils.format(np.linspace(tmp[-1], 0, self.slope_length)))
        )
        del tmp
        return output

    def points_to_plateaus(self, data):
        """
        Generates plateaus for the points inputted

        Parameters
        ----------
        data : torch.tensor
            points for which plateaus are generated

        Returns
        -------
        torch.tensor
            plateaus generated from points as a torch tensor

        Example
        -------
        waveform_mgr = WaveformManager(configs)
        data = (1,1)
        points = torch.rand(data)
        plateaus = waveform_mgr.points_to_pleateaus(points)

        """
        return self.tile(data, 0, self.plateau_length)

    def tile(self, t, dim, n_tile):
        init_dim = t.size(dim)
        repeat_idx = [1] * t.dim()
        repeat_idx[dim] = n_tile
        t = t.repeat(*(repeat_idx))
        order_index = torch.cat([
            init_dim * torch.arange(n_tile, device=t.device, dtype=torch.long)
            + i for i in range(init_dim)
        ])
        return torch.index_select(t, dim, order_index)

    def plateaus_to_waveform(
        self,
        data: torch.Tensor,
        return_pytorch=True
    ) -> Tuple[Union[np.array, torch.Tensor], Union[list[bool], torch.Tensor]]:
        """
        Transform plateau data into full waveform data by adding
        slopes inbetween the plateaus.
        Creates new array and alternates between adding slopes and
        plateaus. Simultaneously makes a mask that indicates the
        positions of the plateaus.
        Will throw error if data size is not multiple of set plateau length
        of the object (self.plateau_length).

        Example
        -------
        >>> manager = WaveformManager({"plateau_length": 2, "slope_length": 2})
        >>> data = torch.tensor([[1], [1], [3], [3]])
        >>> manager.plateaus_to_waveform(data)
        (torch.tensor([[0], [1], [1], [1], [1],
                       [3], [3], [3], [3], [0]]),
        torch.tensor([False, False, True, True, False, False,
                      True, True, False, False])

        In this example we have 2 plateaus of length 2, which is also the
        length of our waveform object. Transforming to waveforms with
        slope length 2 adds a plateau of length 2 inbetween each of the 2
        plateaus.

        Parameters
        ----------
        data : torch.Tensor
            The input data, should consist of sequences of repeated numbers,
            each sequence having the same length of the set plateau length of
            the object.
        return_pytorch : bool, optional
            Indicates whether to return a pytorch tensor (true) or a numpy
            array (false). Default is true.

        Returns
        -------
        output_data : torch.Tensor or np.array
            The plateau data with the added slopes.
        output_mask : list[bool] or torch.Tensor
            The resulting mask - list of booleans with true at plateaus and
            false at slopes (or a 1D tensor).

        Raises
        ------
        AssertionError
            If the lenght of the input data is not a multiple of the plateau
            length of the object.
        """
        # Check input format.
        assert (len(data) % self.plateau_length == 0
                ), f"Length of input data {data.shape} is not multiple of "
        f"plateau length {self.plateau_length}."

        data_size = int(len(data) / self.plateau_length)  # number of plateaus
        input_copy = TorchUtils.get_numpy_from_tensor(
            data)  # numpy copy of input data (numpy linspace works for
        # multidimensional data while torch does not)
        start = 0  # starting position of current plateau in input data

        # Initiate output.
        output_data = np.linspace(0, input_copy[start], self.slope_length)
        output_mask = [False] * self.slope_length

        # Go through all data except last plateau.
        for i in range(data_size - 1):
            end = start + self.plateau_length
            output_mask += [True] * self.plateau_length
            output_data = np.concatenate((output_data, input_copy[start:end]))
            output_mask += [False] * self.slope_length
            output_data = np.concatenate(
                (output_data,
                 np.linspace(input_copy[end - 1], input_copy[end],
                             self.slope_length)))
            start = end

        # Go through last plateau and final slope.
        output_mask += [True] * self.plateau_length
        output_data = np.concatenate((output_data, input_copy[start:]))
        output_mask += [False] * self.slope_length
        output_data = np.concatenate(
            (output_data, np.linspace(input_copy[-1], 0, self.slope_length)))

        if return_pytorch:
            return TorchUtils.get_tensor_from_numpy(
                output_data), TorchUtils.get_tensor_from_list(output_mask,
                                                              data_type=bool)
        else:
            return output_data, output_mask

    def plateaus_to_points(self, data: torch.Tensor) -> torch.Tensor:
        """
        Transform a tensor of plateaus to a tensor of points. This is done by
        reshaping the data such that one dimension is the plateau length,
        then removing that dimension by taking the mean over it.

        Example
        -------
        >>> manager = WaveformManager({"plateau_length": 4, "slope_length": 2})
        >>> data = torch.tensor([[1], [1], [1], [1], [5], [5], [5], [5],
                                 [3], [3], [3], [3]])
        >>> manager.plateaus_to_points(data)
        torch.tensor([[1], [5], [3]])

        In this example we have 3 plateaus of length 4.

        Parameters
        ----------
        data : torch.Tensor
            The input data, should consist of sequences of repeated numbers,
            each sequence having the lenght of the set plateau length of the
            object.

        Returns
        -------
        output : torch.Tensor
            Tensor where every plateau of the input data is represented
            by a single point.

        Raises
        ------
        AssertionError
            If the lenght of the input data is not a multiple of the plateau
            length of the object.
        """
        # Check input format.
        assert (len(data) % self.plateau_length == 0
                ), f"Length of input data {data.shape} is not multiple of "
        f"plateau length {self.plateau_length}."

        data_size = int(len(data) / self.plateau_length)  # number of plateaus

        # Reshape input so that each data point is represented along
        # dimension 0, then take average over dimension 1 to get rid
        # of plateaus.
        output = data.view(data_size, self.plateau_length,
                           data.shape[1]).mean(dim=1)

        # Make the output two-dimensional.
        if len(output.shape) == 1:
            output = output.unsqueeze(dim=1)
        return output

    def waveform_to_points(self, data=torch.Tensor, mask=None) -> torch.Tensor:
        """
        Transform waveform data to point data. First apply a mask to remove
        the slopes, then apply self.plateaus_to_points to get only points.
        If a mask is not given, it will be generated.

        Example
        -------
        >>> manager = WaveformManager({"plateau_length": 1, "slope_length": 2})
        >>> data = torch.tensor([[0], [1], [1], [1], [5], [5], [5], [0]])
        >>> manager.waveform_to_points(data)
        torch.tensor([[1], [5]])

        Parameters
        ----------
        data : torch.Tensor
            Input data in waveform form.
        mask : Sequence[bool], optional
            Provide a mask, by default None.

        Returns
        -------
        self.plateaus_to_points(data[mask])
            A tensor where each data point is represented once.

        Raises
        ------
        AssertionError
            If the lenght of the input data is not a multiple of the plateau
            length of the object.
        """
        if mask is None:
            mask = self.generate_mask(len(data))
        return self.plateaus_to_points(self.waveform_to_plateaus(data, mask))

    def waveform_to_plateaus(self,
                             data: torch.Tensor,
                             mask=None) -> torch.Tensor:
        """
        Go from waveform to only plateaus by removing the slopes.
        Either generate a mask or use a given one.

        Assume input data is infact a waveform (no size assertion).

        Example
        -------
        >>> manager = WaveformManager({"plateau_length": 2, "slope_length": 2})
        >>> data = torch.tensor([[0], [1], [1], [1], [1],
                                 [5], [5], [5], [5], [0]])
        >>> manager.waveform_to_plateaus(data)
        torch.tensor([[1], [1], [5], [5]])

        Parameters
        ----------
        data : torch.Tensor
            Input data in waveform form.
        mask : Sequence[bool], optional
            Provide a mask, by default None

        Returns
        -------
        data[mask] : torch.Tensor
            Tensor with the slopes removed.
        """
        if mask is None:
            mask = self.generate_mask(len(data))
        return data[mask]

    def generate_mask(self, data_size: int) -> torch.Tensor:
        """
        Use self.mask and self.final_mask to make a mask for input
        of given size:
        if there are 3 data points, return
        self.mask * 3 + self.final_mask
        self.mask is [False] * self.slope_length + [True] * self.plateau_length
        self.final_mask is [False] * self.slope_length

        Assume the data size is valid.

        Example
        -------
        >>> configs = {"plateau_length": 2, "slope_length": 1}
        >>> manager = WaveformManager(configs)
        >>> manager.generate_mask(7)
        torch.tensor([False, True, True, False, True, True, False])

        This example has two plateaus of length 2 and 3 slopes of length 1.

        Parameters
        ----------
        data_size : int
            The number of points in the data.

        Returns
        -------
        torch.cat((mask, self.final_mask)) : torch.Tensor
            A mask of the required length.

        """
        repetitions = int(((data_size - self.slope_length) /
                           (self.slope_length + self.plateau_length)))
        mask = self.initial_mask.clone().repeat(repetitions)
        return torch.cat((mask, self.final_mask))


def process_data(waveform_transforms, inputs, targets):
    # Data processing required to apply waveforms to the inputs and pass them onto the GPU if necessary.
    if waveform_transforms is not None:
        inputs, targets = waveform_transforms((inputs, targets))
    if inputs is not None and inputs.device != TorchUtils.get_accelerator_type(
    ):
        inputs = inputs.to(device=TorchUtils.get_accelerator_type())
    if targets is not None and targets.device != TorchUtils.get_accelerator_type(
    ):
        targets = targets.to(device=TorchUtils.get_accelerator_type())

    return inputs, targets
