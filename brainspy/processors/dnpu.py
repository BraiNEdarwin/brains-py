"""
Module for handling the DNPU processor, freezing the control voltages,
splitting and merging the input and control electrode data, handling
the biases of the network.
"""
import collections
from typing import Sequence, Union

import torch
import numpy as np
from torch import nn, Tensor

from brainspy.utils.pytorch import TorchUtils
from brainspy.processors.processor import Processor


class DNPU(nn.Module):
    """
    Class that contains the DNPU processor and its attributes, like the control
    voltage ranges, values of the control electrodes (bias) and the indices
    of the data and control electrodes.

    Attributes
    ----------
    processor : Processor
        The processor, which in turn contains a simulation processor or a
        hardware processor.
    control_low : torch.Tensor
        The lower boundary of the control voltages allowed.
    control_high : torch.Tensor
        The upper boundary of the control voltages allowed.
    bias : torch.Tensor
        The biases of the network (control voltages).
    data_input_indices : torch.Tensor
        The indices of the input electrodes.
    control_indices : torch.Tesnor
        The indices of the control electrodes.
    """
    def __init__(
        self,
        configs: dict,
        info: dict = None,
        model_state_dict: collections.OrderedDict = None,
        processor: Processor = None,
    ):
        """
        Initialize the DNPU, either with a Processor object or an info
        dictionary which is used to create a Processor.

        Parameters
        ----------
        configs : dict
            Configs dictionary of the processor, documented in Processor class.
        info : dict
            Info dictionary of the processor, documented in Processor class.
        model_state_dict : collections.OrderedDict, optional
            The state dictionary for the parameters of the neural network,
            containing the weights and biases. If not provided, the parameters
            will be initialized with random values. By default None.
        processor : Processor, optional
            A processor object, by default None

        Raises
        ------
        Exception
            If neither info dictionary nor processor is provided.
        """
        super(DNPU, self).__init__()
        if processor is not None:
            self.processor = processor
        elif info is not None:
            self.processor = Processor(configs, info, model_state_dict)
        else:
            raise Exception("The DNPU must be initialised either with a "
                            "processor or an info dictionary")
        self._init_electrode_info(configs["input_indices"])
        self._init_dnpu()

    def _init_dnpu(self):
        """
        Freeze parameters of the neural network of the surrogate model and
        initiate the biases.

        Method is called by constructor.
        """
        for (params) in (self.parameters()):
            params.requires_grad = False
        self._init_bias()

    def _init_bias(self):
        """
        Set the biases of the neural network of the surrogate model
        (control voltages).
        Check for negative voltages, set the biases to a random value in the
        control voltage ranges. Declare them as neural network parameters.

        Method is called by constructor (indirectly) and reset method.

        Raise
        -----
        AssertionError
            If negative voltages are detected.
        """
        self.control_low = self.get_control_ranges()[:, 0]
        self.control_high = self.get_control_ranges()[:, 1]

        # check for negative voltages
        assert any(
            self.control_low < 0
        ), "Min. Voltage is assumed to be negative, but value is positive!"
        assert any(
            self.control_high > 0
        ), "Max. Voltage is assumed to be positive, but value is negative!"

        # set the bias to a random value between low and high
        bias = self.control_low + (self.control_high -
                                   self.control_low) * torch.rand(
                                       1,
                                       len(self.control_indices),
                                       dtype=torch.get_default_dtype(),
                                       device=TorchUtils.get_device(),
                                   )

        # declare the bias as neural network parameters so they are affected
        # by backpropagation
        self.bias = nn.Parameter(bias)

    def _init_electrode_info(self, input_indices: Sequence[int]):
        """
        Set input electrode indices and control electrode indices.

        Method is called by constructor.

        Example
        -------
        >>> dnpu.get_activation_electrode_no()
        7
        >>> input_indices = [0, 2]
        >>> dnpu._init_electrode_info()
        >>> dnpu.data_input_indices
        torch.Tensor([0, 2])
        >>> dnpu.control_indices
        torch.Tensor([1, 3, 4, 5, 6])

        Here we have a DNPU with 7 activation electrodes, where the two with
        indices 0 and 2 are set as input electrodes, and the rest become
        control electrodes.

        Parameters
        ----------
        input_indices : Sequence[int]
            Indices of the input electrodes.
        """
        self.data_input_indices = TorchUtils.format(input_indices,
                                                    data_type=torch.int64)
        self.control_indices = np.delete(
            np.arange(self.processor.get_activation_electrode_no()),
            input_indices)
        self.control_indices = TorchUtils.format(self.control_indices,
                                                 data_type=torch.int64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run a forward pass through the processor after merging the input data.

        Parameters
        ----------
        x : torch.Tensor
            Input data.

        Returns
        -------
        torch.Tensor
            Output data.
        """
        merged_data = merge_electrode_data(
            x,
            self.bias.expand(x.size()[0], -1),
            self.data_input_indices,
            self.control_indices,
        )
        return self.processor(merged_data)

    def regularizer(self) -> torch.Tensor:
        """
        Return a penalty term if the value of the bias is outside of the
        interval for the control voltages.

        Example
        -------
        >>> dnpu.control_low
        torch.Tensor([1.0, 5.0])
        >>> dnpu.control_high
        torch.Tensor([3.0, 7.0])
        >>> dnpu.bias
        torch.Tensor([2.5, 8.0])
        >>> dnpu.regularizer()
        torch.Tensor([1.0])

        In this example we have two control electrodes, the first with voltage
        range 1 to 3 and the second 5 to 7. The bias of the network is 2.5 for
        the first electrode and 8 for the second. The first is within the
        boundaries so no penalty is generated from it but the second is
        outside, which means a penalty will be generated, which is equal to
        how far outside the interval the value is. In this case 8 - 7 = 1, so
        the penalty will be 0 + 1 = 1.

        Returns
        -------
        torch.Tensor
            Penalty term >=0.
        """
        return torch.sum(
            torch.relu(self.control_low - self.bias) +
            torch.relu(self.bias - self.control_high))

    def hw_eval(self, arg):
        """
        Set the module to evaluation mode (meaning no training) and set
        the processor to a new one. Check if the control voltage ranges of the
        new one are the same.

        Parameters
        ----------
        arg : Processor or dict
            Either a processor or configs dictionary to be applied to the
            existing one.

        Raises
        ------
        AssertionError
            If control voltages of the new processor are different than the
            ones used for training the DNPU.
        """
        self.eval()
        if isinstance(arg, Processor):
            self.processor = arg
        else:
            self.processor.load_processor(arg)
        assert torch.equal(
            self.control_low.cpu().half(),
            self.processor.get_control_ranges()[:, 0].cpu().half(),
        ), "Low control voltage ranges for the new processor are different "
        "than the control voltage ranges for which the DNPU was trained."
        assert torch.equal(
            self.control_high.cpu().half(),
            self.processor.get_control_ranges()[:, 1].cpu().half(),
        ), "High control voltage ranges for the new processor are different "
        "than the control voltage ranges for which the DNPU was trained."

    def set_control_voltages(self, bias: torch.Tensor):
        """
        Change the control voltages/bias of the network.

        Example
        -------
        >>> dnpu.set_control_voltages(torch.tensor([1.0, 2.0, 3.0, 4.0]))

        Parameters
        ----------
        bias : torch.Tensor
            New value of the bias/control voltage.
            One dimensional tensor.
        """
        with torch.no_grad():
            bias = bias.unsqueeze(dim=0)  # add a dimension 0
            assert (
                self.bias.shape == bias.shape
            ), "Control voltages could not be set due to a shape missmatch "
            "with regard to the ones already in the model."
            self.bias = torch.nn.Parameter(TorchUtils.format(bias))

    def get_control_voltages(self) -> torch.Tensor:
        """
        Get the (next) control voltages/bias of the network, detach it from
        the computational graph.

        Returns
        -------
        torch.Tensor
            Value of the bias/control voltage.
        """
        return next(self.parameters()).detach()

    def get_input_ranges(self) -> torch.Tensor:
        """
        Get the voltage ranges of the input electrodes.

        Returns
        -------
        torch.Tensor
            Input voltage ranges.
        """
        return self.processor.get_voltage_ranges()[self.data_input_indices]

    def get_control_ranges(self) -> torch.Tensor:
        """
        Get the voltage ranges of the control electrodes.

        Returns
        -------
        torch.Tensor
            Control voltage ranges.
        """
        return self.processor.get_voltage_ranges()[self.control_indices]

    def get_clipping_value(self) -> torch.Tensor:
        """
        Get the output clipping/clipping value.

        Returns
        -------
        torch.Tensor
            The output clipping of the processor.
        """
        return self.processor.get_clipping_value()

    def reset(self):
        """
        Reset the bias of the processor.
        """
        del self.bias
        self._init_bias()

    # TODO: Document the need to override the closing of the processor on
    # custom models.
    def close(self):
        """
        Close the processor. For simulation models does nothing.
        """
        self.processor.close()

    def is_hardware(self) -> bool:
        """
        Check if processor is a hardware processor or a simulation processor.

        Returns
        -------
        bool
            True if hardware, False if simulation.
        """
        return self.processor.is_hardware()

    # TODO: Document the need to override the closing of the return of the
    # info dictionary.
    def get_info_dict(self) -> dict:
        """
        Get the info dictionary of the processor.

        Returns
        -------
        dict
            The info dictionary.
        """
        return self.processor.info


def merge_electrode_data(
    inputs,
    control_voltages,
    input_indices: Sequence[int],
    control_voltage_indices,
    use_torch=True,
) -> Union[np.array, Tensor]:
    """
    Merge data from two electrodes with the specified indices for each.
    Need to indicate whether numpy or torch is used. The result will
    have the same type as the input.

    Example
    -------
    >>> inputs = np.array([[1.0, 3.0], [2.0, 4.0]])
    >>> control_voltages = np.array([[5.0, 7.0], [6.0, 8.0]])
    >>> input_indices = [0, 2]
    >>> control_voltage_indices = [3, 1]
    >>> electrodes.merge_electrode_data(
    ...     inputs=inputs,
    ...     control_voltages=control_voltages,
    ...     input_indices=input_indices,
    ...     control_voltage_indices=control_voltage_indices,
    ...     use_torch=False,
    ... )
    np.array([[1.0, 7.0, 3.0, 5.0], [2.0, 8.0, 4.0, 6.0]])

    Merging two arrays of size 2x2, resulting in an array of size 2x4.

    Parameters
    ----------
    inputs: np.array or torch.tensor
        Data for the input electrodes.
    control_voltages: np.array or torch.tensor
        Data for the control electrodes.
    input_indices: iterable of int
        Indices of the input electrodes.
    control_voltage_indices: iterable of int
        Indices of the control electrodes.
    use_torch : boolean
        Indicate whether the data is pytorch tensor (instead of a numpy array)

    Returns
    -------
    result: np.array or torch.tensor
        Array or tensor with merged data.

    """
    result = np.empty(
        (inputs.shape[0], len(input_indices) + len(control_voltage_indices)))
    if use_torch:
        result = TorchUtils.format(result)
    result[:, input_indices] = inputs
    result[:, control_voltage_indices] = control_voltages
    return result
