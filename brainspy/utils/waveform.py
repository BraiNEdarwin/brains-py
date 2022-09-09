"""
This module is part of the utils of brains-py helps managing
the waveforms of the signals sent to and received by the hardware DNPUs.

Data can exist in 3 forms:
-points (e.g. (1, 2, 3))
-plateaus (e.g. (1, 1, 1, 2, 2, 2, 3, 3, 3))
-waveform (e.g (0, 0.5, 1, 1, 1, 1, 1, 1.5, 2, 2, 2, 2, 2, 2.5,
3, 3, 3, 3, 3, 1.5, 0))
A waveform transform is defined by its plateau length and slope length,
in the case above 3 and 3 respectively. There are methods in this module
that define the transformations between these three forms.

The goal of the waveform representation of data is so that it can be applied
to DNPUs without sudden changes in input, so that the hardware is not damaged.
"""
from typing import Union, Tuple, List

import torch
import numpy as np
import warnings
from brainspy.utils.pytorch import TorchUtils


class WaveformManager:
    """
    This class helps managing the waveforms of the signals sent to and
    received by the hardware DNPUs (Dopant Network Processing Units).

    The waveform represents a set of points. Each of the points is represented
    with a slope, a plateau and another slope. The first slope is a line that
    goes from the previous point to the current point value. The plateau
    repeats the same point a specified number of times. The second slope is a
    line that goes from the current point to the next point. The starting and
    ending points are considered zero.

    Attributes
    ----------
    plateau_length : int
        The length of the plateaus of this manager.
    slope_length : int
        The length of the slopes of this manager.
    initial_mask : List[bool]
        A mask that covers one slope and one plateau. False where there is a
        slope, True where there is a plateau.
    final_mask : List[bool]
        A mask that covers one plateau - consists entirely of False.
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
        assert (configs["plateau_length"] is not None
                and type(configs["plateau_length"] == int))
        assert (configs["slope_length"] is not None
                and type(configs["plateau_length"] == int))
        if configs["plateau_length"] == 0:
            warnings.warn("Plateau length is 0")
        if configs["slope_length"] == 0:
            warnings.warn("Slope Length is 0")
        self.plateau_length = configs["plateau_length"]
        self.slope_length = configs["slope_length"]
        self.generate_mask_base()

    def generate_mask_base(self):
        """
        To generate a mask base for the torch tensor based on the slope length
        and plateau_length.

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
        parameter : int
            value that specifies the amplitude which can be in the form of an
            integer or a list
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
        assert (parameter is not None and length is not None)
        assert isinstance(parameter, int)
        return [parameter] * length

    def points_to_waveform(self, data):
        """
        Generates a waveform (voltage input over time) with constant intervals
        of value amplitudes[i] for interval i of length[i].

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
        assert type(
            data) is torch.Tensor, "Data provided is not a pytorch Tensor"
        assert len(
            data.shape
        ) >= 2, "Data requires to be in at least two dimensions (data, electrode_no)"
        data_size = len(data) - 1
        tmp = TorchUtils.to_numpy(data)
        output = TorchUtils.format(np.linspace(0,
                                               tmp[0],
                                               num=self.slope_length,
                                               endpoint=False),
                                   device=data.device,
                                   data_type=data.dtype)
        for i in range(data_size):
            output = torch.cat((output, data[i].repeat(self.plateau_length,
                                                       1)))
            output = torch.cat((
                output,
                TorchUtils.format(np.linspace(tmp[i],
                                              tmp[i + 1],
                                              num=self.slope_length + 1,
                                              endpoint=False)[1:],
                                  device=data.device,
                                  data_type=data.dtype),
            ))
        output = torch.cat((output, data[-1].repeat(self.plateau_length, 1)))
        output = torch.cat(
            (output,
             TorchUtils.format(np.linspace(tmp[-1],
                                           0,
                                           num=self.slope_length + 1)[1:],
                               device=data.device,
                               data_type=data.dtype)))
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
        return data.repeat_interleave(self.plateau_length, dim=0)

    def plateaus_to_waveform(
        self,
        data: torch.Tensor,
        return_pytorch=True
    ) -> Tuple[Union[np.ndarray, torch.Tensor], Union[List[bool],
                                                      torch.Tensor]]:
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
        output_mask : List[bool] or torch.Tensor
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
        input_copy = TorchUtils.to_numpy(
            data)  # numpy copy of input data (numpy linspace works for
        # multidimensional data while torch does not)
        start = 0  # starting position of current plateau in input data

        # Initiate output.
        output_data = np.linspace(0,
                                  input_copy[start],
                                  num=self.slope_length,
                                  endpoint=False)
        output_mask = [False] * self.slope_length

        # Go through all data except last plateau.
        for i in range(data_size - 1):
            end = start + self.plateau_length
            output_mask += [True] * self.plateau_length
            output_data = np.concatenate((output_data, input_copy[start:end]))
            output_mask += [False] * self.slope_length
            output_data = np.concatenate((
                output_data,
                np.linspace(input_copy[end - 1],
                            input_copy[end],
                            num=self.slope_length + 1,
                            endpoint=False)[1:],
            ))
            start = end

        # Go through last plateau and final slope.
        output_mask += [True] * self.plateau_length
        output_data = np.concatenate((output_data, input_copy[start:]))
        output_mask += [False] * self.slope_length
        output_data = np.concatenate(
            (output_data,
             np.linspace(input_copy[-1], 0, num=self.slope_length + 1)[1:]))

        if return_pytorch:
            return (
                TorchUtils.format(output_data,
                                  device=data.device,
                                  data_type=data.dtype),
                TorchUtils.format(output_mask,
                                  device=data.device,
                                  data_type=bool),
            )
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
        data = data.unsqueeze(1)
        data_shape = list(data.shape)
        data_shape[0] = data_size
        data_shape[1] = self.plateau_length
        output = data.view(data_shape).mean(dim=1)

        # Make the output two-dimensional.
        # if len(output.shape) == 1:
        #     output = output.unsqueeze(dim=1)
        return output

    def waveform_to_points(self,
                           data: torch.Tensor,
                           mask=None) -> torch.Tensor:
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
        torch.Tensor
            A tensor where each data point is represented once.

        Raises
        ------
        AssertionError
            If the lenght of the input data is not a multiple of the plateau
            length of the object.
        """
        assert type(
            data) is torch.Tensor, "Data provided is not a pytorch Tensor"
        assert len(
            data.shape
        ) >= 2, "Data requires to be in at least two dimensions (data, electrode_no)"
        print(data.shape)
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
        torch.Tensor
            Tensor with the slopes removed.
        """
        assert type(
            data) is torch.Tensor, "Data provided is not a pytorch Tensor"
        assert len(
            data.shape
        ) >= 2, "Data requires to be in at least two dimensions (data, electrode_no)"
        if mask is None:
            mask = self.generate_mask(len(data)).to(data.device)
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
        torch.Tensor
            A mask of the required length.

        """
        repetitions = int(((data_size - self.slope_length) /
                           (self.slope_length + self.plateau_length)))
        mask = self.initial_mask.clone().repeat(repetitions)
        return torch.cat((mask, self.final_mask))
