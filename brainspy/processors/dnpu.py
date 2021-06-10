import torch
import collections
import numpy as np

from torch import nn, Tensor
from typing import Sequence, Union

from brainspy.utils.pytorch import TorchUtils
from brainspy.processors.processor import Processor

from brainspy.utils.transforms import get_linear_transform_constants


class DNPU(nn.Module):
    """

    Attributes
    ----------
    processor : Processor
        An instance of a Processor class representing either a simulation processor or a
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
        processor: Processor,
        data_input_indices: list,  # Data input electrode indices. It should be a double list (e.g., [[1,2]] or [[1,2],[1,3]])
        forward_pass_type: str = 'vec',  # It can be 'for' in order to do time multiplexing with the same device using a for loop, and 'vec' in order to do time multiplexing with vectorisation. )
    ):
        super(DNPU, self).__init__()
        self.processor = processor
        self.forward_pass_type = forward_pass_type
        self.init_electrode_info(data_input_indices)
        self._init_learnable_parameters()
        self.set_forward_pass(forward_pass_type)
        self.input_transform = False
        self.input_clip = False
        
    def set_forward_pass(self, forward_pass_type):
        if forward_pass_type == 'vec':
            self.forward_pass = self.forward_vec
        elif forward_pass_type == 'for':
            self.forward_pass = self.forward_for
        else:
            assert False, "Dnpu type not recognised. It should be either 'single', 'for' or 'vec'"
            # TODO: Change the assestion for raising an exception

    def init_node_no(self, data_input_indices):
        aux = TorchUtils.format(data_input_indices)
        if len(aux.shape) == 1:
            return 1
        else:
            return len(aux)

    def init_activation_electrode_no(self, data_input_indices):
        aux = TorchUtils.format(data_input_indices)
        input_data_electrode_no = len(aux[0])
        control_electrode_no = self.processor.get_activation_electrode_no() - input_data_electrode_no
        return input_data_electrode_no, control_electrode_no

    def get_node_no(self):
        return self.node_no

    def init_electrode_info(self, data_input_indices: Sequence[int]):
        """
        Set the input data electrode indices and control electrode indices,
        as well as the data input and control voltage ranges allowed according
        to the assigned processor electrodes.

        Method is called by the constructor.

        Example
        -------
        >>> dnpu.get_activation_electrode_no()
        7
        >>> input_indices = [0, 2]
        >>> dnpu._init_electrode_info(input_indices)
        >>> dnpu.data_input_indices
        torch.Tensor([0, 2])
        >>> dnpu.control_indices
        torch.Tensor([1, 3, 4, 5, 6])

        Here we have a DNPU with 7 activation electrodes, where the two with
        indices 0 and 2 are set as input electrodes, and the rest become
        control electrodes.

        Parameters
        ----------
        data_input_indices : Sequence[int]
            Indices of the input electrodes.
        """
        self.node_no = self.init_node_no(data_input_indices)
        self.data_input_electrode_no, self.control_electrode_no = self.init_activation_electrode_no(data_input_indices)
        voltage_ranges = self.processor.processor.get_voltage_ranges()

        # Define data input voltage ranges
        # TODO: Add to documentation. data input ranges are defined as follows: (node_no, electrode_no, 2) where last 2 is for min and max
        # Define data input electrode indices
        self.data_input_indices = TorchUtils.format(
            data_input_indices, data_type=torch.int64
        )
        assert len(self.data_input_indices.shape) == 2, "Please revise the format in which data input indices has been passed to the DNPU. Data input indices should be represented with two dimensions (number of DNPU nodes, number of data input electrodes) (e.g., [[1,2]] or [[1,2],[1,3]], data input indices CANNOT be represented as just [1,2]. )"
        self.data_input_ranges = torch.stack([voltage_ranges[i] for i in data_input_indices])

        # Define control voltage ranges
        activation_electrode_indices = np.arange(self.processor.get_activation_electrode_no())
        control_list = [np.delete(activation_electrode_indices, i) for i in data_input_indices]
        # TODO: Add to documentation. control ranges are defined as follows: (node_no, electrode_no, 2) where last 2 is for min and max
        self.control_ranges = TorchUtils.format(torch.stack([voltage_ranges[i] for i in control_list]))

        # Define control electrode indices
        self.control_indices = TorchUtils.format(control_list, data_type=torch.int64)  # IndexError: tensors used as indices must be long, byte or bool tensors

    def get_data_input_electrode_no(self):
        return self.data_input_electrode_no

    def get_control_electrode_no(self):
        return self.control_electrode_no

    # Allows to constraint the control voltages to their corresponding maximum and minimum values.
    # Can be used after loss.backward() and optimizer.step() to clip out those control voltages that are outside their ranges.
    def constraint_control_voltages(self):
        self.bias = torch.nn.Parameter(torch.max(torch.min(self.bias, self.get_control_ranges().T[1].T), self.get_control_ranges().T[0].T))

    # Returns a random single value of a control voltage within a specified range.
    # Control voltage range = [min,max]
    def sample_controls(self):
        random_voltages = TorchUtils.format(torch.rand_like(self.control_indices.float())).T
        range_size = (self.control_ranges.T[1] - self.control_ranges.T[0])
        range_base = self.control_ranges.T[0]
        return ((range_size * random_voltages) + range_base).T

    def _init_learnable_parameters(self):
        # Freeze parameters of the neural network of the surrogate model (in case of having a simulation processor)
        for params in self.parameters():
            params.requires_grad = False
        self._init_bias()

    def _init_bias(self):
        """
        Sets the intial random values for the control voltages of the DNPU, and declares them
        as learnable pytorch parameters. The initial random values are specified within the range
        that the processor has defined by default for a specific electrode.

        Method is called by constructor (indirectly) and reset method.

        Raise
        -----
        AssertionError
            If negative voltages are detected.
        """
        self.bias = nn.Parameter(self.sample_controls())

    # node_index: For time multiplexing only. Is the index of the processor to which the data will be sent.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ 
        Run a forward pass through the processor after merging the input data voltages with the control
        voltages.

        Parameters
        ----------
        x : torch.Tensor
            Input data.

        Returns
        -------
        torch.Tensor
            Output data.
        """
        return self.forward_pass(x)

    def forward_single(self, x, control_voltages, data_input_indices, control_indices):
        merged_data = merge_electrode_data(
            x,
            control_voltages.expand(x.size()[0], -1),
            data_input_indices,
            control_indices,
        )
        return self.processor(merged_data)

    def forward_for(self, x):
        # Cut input values to force linear transform or force being between a voltage range.
        x = self.clamp_input(x)

        # Apply a linear transformation from raw data to the voltage ranges of the dnpu.
        if self.transform_to_voltage:
            x = (self.scale.flatten() * x) + self.offset.flatten()
        assert (
            x.shape[-1] == self.data_input_indices.numel()
        ), f"size mismatch: data is {x.shape}, DNPU_Layer expecting {self.processor.inputs_list.numel()}"
        outputs = [
            self.forward_single(
                node_x,
                self.bias[node_index],
                self.data_input_indices[node_index],
                self.control_indices[node_index],
            )
            for node_index, node_x in enumerate(self.get_node_input_data(x))
        ]

        return torch.cat(outputs, dim=1)

    def forward_vec(self, x):
        batch_size = x.size(0)
        data_input_shape = list(self.data_input_indices.shape)
        data_input_shape.insert(0, batch_size)
        bias_shape = list(self.bias.shape)
        bias_shape.insert(0, batch_size)
        # Reshape input and expand controls
        x = x.reshape(data_input_shape)
        last_dim = len(x.size()) - 1
        controls = self.bias.expand(bias_shape)

        # Cut input values to force linear transform or force being between a voltage range.
        x = self.clamp_input(x)

        # Apply a linear transformation from raw data to the voltage ranges of the dnpu.
        if self.transform_to_voltage:
            x = (self.scale * x) + self.offset

        # Expand indices according to batch size
        input_indices = self.data_input_indices.expand(data_input_shape)
        control_indices = self.control_indices.expand(bias_shape)

        # Create input data and order it according to the indices
        indices = torch.argsort(torch.cat((input_indices, control_indices), dim=last_dim),dim=last_dim)
        data = torch.cat((x, controls), dim=last_dim)
        data = torch.gather(data, last_dim, indices)

        # pass data through the processor
        return self.processor(data).squeeze(-1)  # * self.node.amplification

    # Strict defines if the input is going to be clipped before doing the linear transformation in order to ensure that the transformation is correct
    # Input range can be simply a [min,max] values for the raw input data, which will be 
    def add_input_transform(self, input_range, strict=True):
        self.input_transform = True
        self.input_clip = strict
        input_range = TorchUtils.format(input_range)
        if input_range.shape != self.data_input_ranges.shape:
            input_range = input_range.expand_as(self.data_input_ranges)
        self.raw_input_range = input_range
        self.transform_to_voltage = True
        scale, offset = get_linear_transform_constants(self.data_input_ranges.T[0].T, self.data_input_ranges.T[1].T, input_range.T[0].T, input_range.T[1].T)
        self.scale = scale
        self.offset = offset

    def remove_input_transform(self):
        self.input_transform = False
        del self.amplitude
        del self.offset

    def clamp_input(self, x):
        if self.input_clip:
            x = torch.max(torch.min(x, self.raw_input_range[:, :, 1]), self.raw_input_range[:, :, 0])
        return x

    def get_node_input_data(self, x):
        i = 0
        while i + self.data_input_indices.shape[-1] <= x.shape[-1]:
            yield x[:, i: i + self.data_input_indices.shape[-1]]
            i += self.data_input_indices.shape[-1]

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
        control_voltages = self.get_control_voltages().T
        control_ranges = self.get_control_ranges().T
        return torch.sum(torch.relu(control_ranges[0] - control_voltages) + torch.relu(control_voltages - control_ranges[1]) )

    # @TODO: Update documentation
    def hw_eval(self, configs: dict, info: dict, model_state_dict: collections.OrderedDict = None,):
        """
        It helps setting the DNPU to evaluation mode. While training happens in simulation, evaluation
        happens in hardware. This function sets the nn.Module to evaluation mode (meaning no training) and swaps
        the processor (typically to hardware, although it also supports to do it for hardware_debug or simulation). 
        Checks if the voltage ranges of the new processor are the same as the ones of the old.

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
        old_ranges = self.processor.get_voltage_ranges().cpu().half().clone()
        self.processor.swap(configs, info, model_state_dict)
        new_ranges = self.processor.get_voltage_ranges().cpu().half().clone()
        assert torch.equal(old_ranges, new_ranges), "Voltage ranges for the new processor are different "
        "than the control voltage ranges for which the DNPU was trained."
        del old_ranges
        del new_ranges
        
    # @TODO: Update documentation
    def sw_train(self, configs: dict, info: dict, model_state_dict: collections.OrderedDict = None,):
        """
        It helps swap the DNPU to training mode. While evaluation happens in hardware, training
        happens in software. This function sets the nn.Module to training mode and swaps
        the processor (typically to software).
        Checks if the voltage ranges of the new processor are the same as the ones of the old.

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
        self.train()
        old_ranges = self.processor.get_voltage_ranges().cpu().half().clone()
        self.processor.swap(configs, info, model_state_dict)
        new_ranges = self.processor.get_voltage_ranges().cpu().half().clone()
        assert torch.equal(old_ranges, new_ranges), "Voltage ranges for the new processor are different "
        "than the control voltage ranges for which the DNPU was trained."
        del old_ranges
        del new_ranges

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
        return self.bias.detach()

    def get_input_ranges(self) -> torch.Tensor:
        """
        Get the voltage ranges of the input electrodes.
        It has a shape of (dnpu_no, electrode_no, 2), where the last dimension is
        0 for the minimum value of the range and 1 for the maximum value of the range.
        Returns
        -------
        torch.Tensor
            Input voltage ranges.
        """
        return self.data_input_ranges

    def get_control_ranges(self) -> torch.Tensor:
        """
        Get the voltage ranges of the control electrodes.
        It has a shape of (dnpu_no, electrode_no, 2), where the last dimension is
        0 for the minimum value of the range and 1 for the maximum value of the range.
        Returns
        -------
        torch.Tensor
            Control voltage ranges.
        """
        return self.control_ranges

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
        Close the processor. For simulation models, it does nothing. For hardware models it closes the drivers.
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

    def format_targets(self, x: torch.Tensor) -> torch.Tensor:
        return self.processor.format_targets(x)

def merge_electrode_data(
    input_data,
    control_data,
    input_data_indices: Sequence[int],
    control_indices,
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

    Returns
    -------
    result: np.array or torch.tensor
        Array or tensor with merged data.

    """
    assert (input_data.dtype == control_data.dtype and input_data.device == control_data.device), "Input data voltages and control voltages have a different data type or are in a different device (CUDA or CPU). "
    result = torch.empty(
        (input_data.shape[0], len(input_data_indices) + len(control_indices)), device=TorchUtils.get_device())
    result[:, input_data_indices] = input_data
    result[:, control_indices] = control_data
    return result
