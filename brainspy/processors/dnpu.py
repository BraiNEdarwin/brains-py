from random import random
import torch
import collections
import numpy as np

from torch import nn, Tensor
from typing import Sequence, Union
from brainspy.processors.processor import Processor
from brainspy.utils.performance import data

from brainspy.utils.transforms import get_linear_transform_constants

import warnings


class DNPU(nn.Module):
    """
    This class enables to declare the control voltages as learnable parameters in order to train a
    surrogate model to find optimal control voltage values for a particular problem. It is a child
    of an nn.Module class that contains a processor. Therefore, the solutions found for control
    voltages using a surrogate model, can be seamlessly applied to hardware DNPUs. The class also
    enables to declare more than one internal DNPU node, that will act as a single layer of the
    specified node number, in time-multiplexing of the same model.
    """
    def __init__(self,
                 processor: Processor,
                 data_input_indices: list,
                 forward_pass_type: str = 'vec'):
        """
        Initialises the super class and makes a separation between those electrodes that are data
        input and those that are control.

        Attributes
        ----------
        processor : Processor
            An instance of a Processor class representing either a simulation processor or a
            hardware processor.

        data_input_indices: list
            Specifies which electrodes are going to be used for inputing data. The reminder of the
            activation electrodes will be automatically selected as control electrodes. The list
            should have the following shape (dnpu_node_no,data_input_electrode_no). The minimum
            dnpu_node_no should be 1, e.g., data_input_indices = [[1,2]]. When specifying more than
            one dnpu node in the list, the module will simulate, in time-multiplexing,
            as if there was a layer of DNPU devices. Fore example, for an 8 electrode DNPU device
            with a single readout electrode and 7 activation electrodes, when
            data_input_indices = [[1,2],[1,3],[3,4]], it will be considered that there are 3 DNPU
            devices, where the first DNPU device will use the data input electrodes
            1 and 2, the second DNPU device will use data input electrodes 1 and 3 and the third
            DNPU device will use data input electrodes 3 and 4. Also, the first DNPU device will
            have electrodes 0, 3, 4, 5, and 6 defined as control electrodes. The second DNPU device
            will have electrodes 0,2,4,5, and 6 defined as control electrodes. The third DNPU device
            will have electrodes 0,1,2,5, and 6 defined as control electrodes. More information
            about what activation, readout, data input and control electrodes are can be found at
            the wiki: https://github.com/BraiNEdarwin/brains-py/wiki/A.-Introduction

        forward_pass_type : str
            It indicates if the forward pass for more than one DNPU devices on time-multiplexing
            will be executed using vectorisation or a for loop. The available options are 'vec' or
            'for'. By default it uses the vectorised version.
        """
        super(DNPU, self).__init__()
        assert isinstance(
            processor, Processor
        ), "Processor should be an instance of brainspy.processors.processor.Processor"
        self.processor = processor

        self.init_electrode_info(data_input_indices)
        self._init_learnable_parameters()

        self.set_forward_pass(forward_pass_type)
        self.input_transform = False
        self.input_clip = False

    def set_forward_pass(self, forward_pass_type: str):
        """
        Sets the type of forward pass that is going to be used.

        Attributes
        ----------
        forward_pass_type : str
            It indicates if the forward pass for more than one DNPU devices on time-multiplexing
            will be executed using vectorisation or a for loop. The available options are 'vec' or
            'for'. By default it uses the vectorised version.
        """
        assert type(forward_pass_type) is str and (
            forward_pass_type == 'vec' or forward_pass_type == 'for'
        ), "Forward pass type not recognised. It should be either 'for' or 'vec'"
        self.forward_pass_type = forward_pass_type

        if forward_pass_type == 'vec':
            self.forward_pass = self.forward_vec
            self.clip_input = self.clip_input_vec
        else:
            self.forward_pass = self.forward_for
            self.clip_input = self.clip_input_for

    def init_activation_electrode_no(self):
        """
        It counts how many control and data input electrodes are
        going to be declared using the data_input_indices variable.

        Attributes
        ----------
        data_input_indices: Sequence[int]
            Specifies which electrodes are going to be used for inputing data. The reminder of the
            activation electrodes will be automatically selected as control electrodes. The list
            should have the following shape (dnpu_node_no,data_input_electrode_no). The minimum
            dnpu_node_no should be 1, e.g., data_input_indices = [[1,2]]. When specifying more than
            one dnpu node in the list, the module will simulate, in time-multiplexing,
            as if there was a layer of DNPU devices. Fore example, for an 8 electrode DNPU device
            with a single readout electrode and 7 activation electrodes, when
            data_input_indices = [[1,2],[1,3],[3,4]], it will be considered that there are 3 DNPU
            devices, where the first DNPU device will use the data input electrodes
            1 and 2, the second DNPU device will use data input electrodes 1 and 3 and the third
            DNPU device will use data input electrodes 3 and 4. Also, the first DNPU device will
            have electrodes 0, 3, 4, 5, and 6 defined as control electrodes. The second DNPU device
            will have electrodes 0,2,4,5, and 6 defined as control electrodes. The third DNPU device
            will have electrodes 0,1,2,5, and 6 defined as control electrodes. More information
            about what activation, readout, data input and control electrodes are can be found at
            the wiki: https://github.com/BraiNEdarwin/brains-py/wiki/A.-Introduction

        Returns
        -------
        input_data_electrode_no - int
            Number of data input electrodes.
        control_electrode_no - int
            Number of control electrodes.

        """
        input_data_electrode_no = len(self.data_input_indices[0])
        control_electrode_no = self.processor.get_activation_electrode_no(
        ) - input_data_electrode_no
        return input_data_electrode_no, control_electrode_no

    def get_node_no(self):
        """
        It counts how many control and data input electrodes are
        going to be declared using the data_input_indices variable.
        """
        return self.node_no

    def init_electrode_info(self, data_input_indices: Sequence[int]):
        """
        Set the input data electrode indices and control electrode indices,
        as well as the data input and control voltage ranges allowed according
        to the assigned processor electrodes. The voltage ranges are defined with
        the following shape: (node_no, electrode_no, 2), where the last dimension
        correspond to the minimum and maximum of the range.

        Method is called by the constructor.

        Example
        -------
        >>> dnpu.get_activation_electrode_no()
        7
        >>> input_indices = [[0, 2]]
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
        assert type(
            data_input_indices
        ) is list, "Data input indices should be provided as a list."
        temp = torch.tensor(data_input_indices)
        assert len(temp.shape) == 2 and temp.shape[0] >= 1 and temp.shape[
            1] <= self.processor.get_activation_electrode_no() and (
                temp < self.processor.get_activation_electrode_no()
            ).all().item() and torch.tensor([
                (e.unique().shape == e.shape) for e in temp
            ]).all().item(), (
                "Please revise the format in which data input indices has been passed to the DNPU. "
                +
                "Data input indices should be represented with two dimensions (number of DNPU nodes, "
                +
                "number of data input electrodes) (e.g., [[1,2]] or [[1,2],[1,3]], data input indices"
                + "CANNOT be represented as just [1,2]. )")
        del temp

        self.register_buffer(
            "data_input_indices",
            torch.tensor(data_input_indices, dtype=torch.long))
        self.node_no = len(self.data_input_indices)
        self.data_input_electrode_no, self.control_electrode_no = self.init_activation_electrode_no(
        )
        voltage_ranges = self.processor.processor.get_voltage_ranges()

        # Define data input voltage ranges
        # TODO: Add to documentation. data input ranges are defined as follows: (node_no,
        # electrode_no, 2) where last 2 is for min and max
        # Define data input electrode indices

        self.register_buffer(
            "data_input_ranges",
            torch.stack([voltage_ranges[i] for i in data_input_indices]))
        # Define control voltage ranges
        activation_electrode_indices = np.arange(
            self.processor.get_activation_electrode_no())
        control_list = [
            np.delete(activation_electrode_indices, i)
            for i in data_input_indices
        ]
        # TODO: Add to documentation. control ranges are defined as follows: (node_no, electrode_no,
        # 2) where last 2 is for min and max
        # self.control_ranges = XX
        self.register_buffer(
            "control_ranges",
            torch.stack([voltage_ranges[i] for i in control_list]))

        self.register_buffer(
            "control_indices",
            torch.tensor(np.array(control_list), dtype=torch.long))
        # # Define control electrode indices
        # self.control_indices = TorchUtils.format(
        #     control_list, data_type=torch.int64
        # )  # IndexError: tensors used as indices must be long, byte or bool tensors

    def get_data_input_electrode_no(self):
        """
        Returns how many data input electrodes are used per DNPU node inside the module.

        Returns
        -------
        data_input_electrode_no - int
            Number of data input electrodes.

        """
        return self.data_input_electrode_no

    def get_control_electrode_no(self):
        """
        Returns how many control electrodes are used per DNPU node inside the module.

        Returns
        -------
        control_electrode_no - int
            Number of control electrodes.

        """
        return self.control_electrode_no

    def constraint_control_voltages(self):
        """
        Allows to constraint the control voltages to their corresponding maximum and minimum values.
        Can be used after loss.backward() and optimizer.step() to clip out those control voltages
        that are outside their ranges.

        Example
        -------
        [...]
        loss.backward()
        optimizer.step()
        model.constraint_weights()
        [...]
        """
        if torch.__version__ >= '1.11.0':
            max_ranges = self.get_control_ranges().permute(
                *torch.arange(self.get_control_ranges().ndim - 1, -1, -1))[1]
            max_ranges = max_ranges.permute(
                *torch.arange(max_ranges.ndim - 1, -1, -1))
            min_ranges = self.get_control_ranges().permute(
                *torch.arange(self.get_control_ranges().ndim - 1, -1, -1))[0]
            min_ranges = min_ranges.permute(
                *torch.arange(min_ranges.ndim - 1, -1, -1))
            self.control_voltages.data = torch.max(
                torch.min(self.control_voltages, max_ranges), min_ranges)

        else:
            self.control_voltages.data = torch.max(
                torch.min(self.control_voltages,
                          self.get_control_ranges().T[1].T),
                self.get_control_ranges().T[0].T)

    # Returns a random single value of a control voltage within a specified range.
    # Control voltage range = [min,max]
    def sample_controls(self):
        """
        Returns a random single value of a control voltage, in the shape of
        (node_no, control_electrode_no), within a specified control voltage range.
        The control voltage ranges are automatically taken from the electrode position.
        Each electrode might have a different voltage range, and this is defined during
        the data gathering process of the surrogate model generator.

        Example
        -------
        [...]
        loss.backward()
        optimizer.step()
        model.constraint_weights()
        [...]
        """

        if torch.__version__ >= '1.11.0':
            random_voltages = torch.rand(self.control_indices.shape,
                                         device=self.control_ranges.device)
            random_voltages = random_voltages.permute(
                *torch.arange(random_voltages.ndim - 1, -1, -1))
            range_base = self.control_ranges.permute(
                *torch.arange(self.control_ranges.ndim - 1, -1, -1))[0]
            range_size = (self.control_ranges.permute(
                *torch.arange(self.control_ranges.ndim - 1, -1, -1))[1] -
                          range_base)
            result = ((range_size * random_voltages) + range_base)
            result = result.permute(*torch.arange(result.ndim - 1, -1, -1))

        else:
            random_voltages = torch.rand(self.control_indices.shape,
                                         device=self.control_ranges.device).T
            range_base = self.control_ranges.T[0]
            range_size = (self.control_ranges.T[1] - range_base)
            result = ((range_size * random_voltages) + range_base).T
        return result

    def _init_learnable_parameters(self):
        """
        Freezes the parameters of the neural network model of the surrogate model and
        initialises the control voltages as learnable parameters of the model.

        """
        for params in self.parameters():
            params.requires_grad = False
        self._init_control_voltages()

    def _init_control_voltages(self):
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
        self.control_voltages = nn.Parameter(self.sample_controls())

    # node_index: For time multiplexing only.
    # Is the index of the processor to which the data will be sent.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run a forward pass through the processor after merging the input data
        voltages with the control voltages.

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

    def forward_single(self, x: torch.Tensor, control_voltages: torch.Tensor,
                       data_input_indices: torch.Tensor,
                       control_indices: torch.Tensor):
        """
        Run a forward pass through the processor for a single DNPU node.

        Parameters
        ----------
        x : torch.Tensor
            Input data in the shape of (batch_size, data_input_electrode_no), where
            data_input_electrode_no is the number of data input electrodes for a
            single DNPU inside the time-multiplexing layer.

        control_voltages : torch.Tensor
            Control voltage values that will be sent to the control electrodes.

        data_input_indices : torch.Tensor [int64]
            Indices of the data input electrodes.

        control_indices : torch.Tensor [int64]
            Indices of the control electrodes.

        Returns
        -------
        torch.Tensor
            Output data in the shape of (batch_size, readout_electrode_no)
        """
        merged_data = merge_electrode_data(
            x,
            control_voltages.expand(x.size()[0], -1),
            data_input_indices,
            control_indices,
        )
        return self.processor(merged_data)

    def forward_for(self, x):
        """
        Run a forward pass through all of the DNPU nodes using a for loop.

        Parameters
        ----------
        x : torch.Tensor
            Input data in the shape of (batch_size, total_data_input_electrode_no), where
            total_data_input_electrode_no is the number of all data input electrodes from
            all the nodes used in the time-multiplexing layer.

        Returns
        -------
        torch.Tensor
            Output data in the shape of (batch_size, readout_electrode_no)
        """
        # Cut input values to force linear transform or force being between a voltage range.
        x = self.clip_input(x)

        # Apply a linear transformation from raw data to the voltage ranges of the dnpu.
        if self.input_transform:
            x = (self.scale.flatten() * x) + self.offset.flatten()
        assert (x.shape[-1] == self.data_input_indices.numel()), (
            f"size mismatch: data is {x.shape}," +
            f"DNPU_Layer expecting {self.data_input_indices.numel()}")
        outputs = [
            self.forward_single(
                node_x,
                self.control_voltages[node_index],
                self.data_input_indices[node_index],
                self.control_indices[node_index],
            ) for node_index, node_x in enumerate(self.get_node_input_data(x))
        ]

        return torch.cat(outputs, dim=1)

    def forward_vec(self, x):
        """
        Run a forward pass through all of the DNPU nodes using by vectorising the input into
        (batch_size, dnpu_node_no, activation_electrode_no).

        Parameters
        ----------
        x : torch.Tensor
            Input data in the shape of (batch_size, total_data_input_electrode_no), where
            total_data_input_electrode_no is the number of all data input electrodes from
            all the nodes used in the time-multiplexing layer.

        Returns
        -------
        torch.Tensor
            Output data in the shape of (batch_size, readout_electrode_no)
        """
        batch_size = x.size(0)
        data_input_shape = list(self.data_input_indices.shape)
        data_input_shape.insert(0, batch_size)
        control_voltages_shape = list(self.control_voltages.shape)
        control_voltages_shape.insert(0, batch_size)
        # Reshape input and expand controls
        x = x.reshape(data_input_shape)
        last_dim = len(x.size()) - 1
        controls = self.control_voltages.expand(control_voltages_shape)

        # Cut input values to force linear transform or force being between a voltage range.
        x = self.clip_input(x)

        # Apply a linear transformation from raw data to the voltage ranges of the dnpu.
        if self.input_transform:
            x = (self.scale.to(x.device) * x) + self.offset.to(x.device)

        # Expand indices according to batch size
        input_indices = self.data_input_indices.expand(data_input_shape)
        control_indices = self.control_indices.expand(control_voltages_shape)

        # Create input data and order it according to the indices
        indices = torch.argsort(torch.cat((input_indices, control_indices),
                                          dim=last_dim),
                                dim=last_dim)
        data = torch.cat((x, controls.to(x.device)), dim=last_dim)
        data = torch.gather(data, last_dim, indices.to(x.device))

        # pass data through the processor
        if self.processor.is_hardware():
            original_data_shape = data.shape
            data = data.reshape(
                original_data_shape[0] * original_data_shape[1], -1)
            result = self.processor(data)
            if not self.processor.average_plateaus:
                result = result.reshape(
                    int(self.processor.waveform_mgr.plateau_length *
                        original_data_shape[0]), original_data_shape[1])
            return result
        else:
            return self.processor(data).squeeze(
                -1)  # * self.node.amplification

    def add_input_transform(self, input_range: list, strict: bool = True):
        """
        Adds a linear transformation required to convert the input into the input electrode
        voltage ranges. It automatically calculates the input electrode voltage ranges for a
        particular DNPU according to the voltage ranges it was trained with. It is used typically
        to perform a current to voltage transformation, but it can also be applied for transforming
        the raw values from a dataset into voltages. The application of the input transformation
        occurs when the data has been reshaped into
        [Batch_size, window_no, in_chanels, node_no, input_electrode_no]. This function
        has to be called from outside the module, after its initialisation.

        Attributes:
        input_range : list
            The range that the original raw input data is going to have. It can be specified with
            two values [min, max], representing the minimum and maximum values that the input data
            is expected to have. In this case, the linear transformation will be adapted to the
            length of the input dimension automatically. E.g. input_range = [0,1].
            It can also be specified for different minimum and maximum value ranges per electrode.
            In this case, the list has to be specified with the same shape as the input_range
            variable of the class. This can be obtained by calling the get_input_ranges method.
        strict : boolean
            Defines if the input is going to be clipped before doing the linear transformation in
            order to ensure that the transformation is correct.
        """
        self.input_transform = True
        self.input_clip = strict
        input_range = torch.tensor(input_range,
                                   dtype=self.data_input_ranges.dtype,
                                   device=self.data_input_ranges.device)
        if input_range.shape != self.data_input_ranges.shape:
            input_range = input_range.expand_as(
                self.data_input_ranges).detach().clone()
        self.register_buffer("raw_input_range", input_range.detach().clone())
        if torch.__version__ >= '1.11.0':
            scale, offset = get_linear_transform_constants(
                self.data_input_ranges.permute(
                    *torch.arange(self.data_input_ranges.ndim -
                                  1, -1, -1))[0].T,
                self.data_input_ranges.permute(
                    *torch.arange(self.data_input_ranges.ndim -
                                  1, -1, -1))[1].T,
                input_range.permute(*torch.arange(input_range.ndim -
                                                  1, -1, -1))[0].T,
                input_range.permute(*torch.arange(input_range.ndim -
                                                  1, -1, -1))[1].T)
        else:
            scale, offset = get_linear_transform_constants(
                self.data_input_ranges.T[0].T, self.data_input_ranges.T[1].T,
                input_range.T[0].T, input_range.T[1].T)
        if scale.unique().shape[0] and offset.unique().shape[0] == 1:
            self.register_buffer("scale", scale.unique())
            self.register_buffer("offset", offset.unique())
            self.unique_transform = True
        else:
            self.register_buffer("scale", scale)
            self.register_buffer("offset", offset)
            self.unique_transform = False

    def remove_input_transform(self):
        """
        Removes the usage of a input transfomration before sending the data to the DNPUs.
        """
        self.input_transform = False
        self.input_clip = False
        if "raw_input_range" in dir(self):
            del self.raw_input_range
        if "scale" in dir(self):
            del self.scale
        if "offset" in dir(self):
            del self.offset

    def clip_input_vec(self, x):
        """
        Clips the input. To be used only during the vectorised forward pass type.

        Parameters
        ----------
        x : torch.Tensor
            Input data that might contain values above or below the specified minimum and
            maximum ranges.

        Returns
        -------
        torch.Tensor
            Clipped data that is assured to be within the specified minimum and maximum ranges.
        """
        if self.input_clip:
            self.raw_input_range = self.raw_input_range.to(x.device)
            x = torch.max(torch.min(x, self.raw_input_range[:, :, 1]),
                          self.raw_input_range[:, :, 0])
        return x

    def clip_input_for(self, x):
        """
        Clips the input. To be used only during the for forward pass type.

        Parameters
        ----------
        x : torch.Tensor
            Input data that might contain values above or below the specified minimum and
            maximum ranges.

        Returns
        -------
        torch.Tensor
            Clipped data that is assured to be within the specified minimum and maximum ranges.
        """
        if self.input_clip:
            self.raw_input_range.to(x.device)
            x = torch.max(torch.min(x, self.raw_input_range[:, :1].flatten()),
                          self.raw_input_range[:, :, 0].flatten())
        return x

    def get_node_input_data(self, x):
        """
        Returns the input data from a particular DNPU node, one by one, from the input data
        of all DNPU nodes.

        Parameters
        ----------
        x : torch.Tensor
            Input data tensor containing the input data per electrode for all DNPU nodes.

        Returns
        -------
        torch.Tensor
            Input data for a particular DNPU node.
        """
        i = 0
        while i + self.data_input_indices.shape[-1] <= x.shape[-1]:
            yield x[:, i:i + self.data_input_indices.shape[-1]]
            i += self.data_input_indices.shape[-1]

    # def refresh_after_processor_swap():
    #     pass

    def regularizer(self) -> torch.Tensor:
        """
        Return a penalty term if the value of the control_voltages is outside of the
        interval for the control voltages.

        Example
        -------
        >>> dnpu.control_low
        torch.Tensor([1.0, 5.0])
        >>> dnpu.control_high
        torch.Tensor([3.0, 7.0])
        >>> dnpu.control_voltages
        torch.Tensor([2.5, 8.0])
        >>> dnpu.regularizer()
        torch.Tensor([1.0])

        In this example we have two control electrodes, the first with voltage
        range 1 to 3 and the second 5 to 7. The control_voltages of the network is 2.5 for
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
        if torch.__version__ >= '1.11.0':
            control_voltages = self.get_control_voltages().permute(
                *torch.arange(self.get_control_voltages().ndim - 1, -1, -1))
            control_ranges = self.get_control_ranges().permute(
                *torch.arange(self.get_control_ranges().ndim - 1, -1, -1))
        else:
            control_voltages = self.get_control_voltages().T
            control_ranges = self.get_control_ranges().T
        return torch.sum(
            torch.relu(control_ranges[0] - control_voltages) +
            torch.relu(control_voltages - control_ranges[1]))

    # @TODO: Update documentation
    def hw_eval(
        self,
        configs: dict,
        data_input_indices: list = None,
    ):
        """
        It helps setting the DNPU to evaluation mode. While training happens in simulation,
        evaluation happens in hardware. This function sets the nn.Module to evaluation mode
        (meaning no training) and swaps the processor (typically to hardware, although it also
        supports to do it for hardware_debug or simulation). Checks if the voltage ranges of the
        new processor are the same as the ones of the old.

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
        assert type(
            configs
        ) is dict, "Configs should be a dictionary. Check Processor for the information that the dictionary should have."
        assert data_input_indices is None or type(
            data_input_indices
        ) is list, "Data input indices should be None or a list with shape (DNPU node number, electrode_no)"
        self.eval()
        old_ranges = self.processor.get_voltage_ranges().cpu().half().clone()
        self.processor.swap(configs, self.get_info_dict())
        if data_input_indices is not None:
            self.init_electrode_info(data_input_indices)
        new_ranges = self.processor.get_voltage_ranges().cpu().half().clone()
        if not torch.equal(old_ranges, new_ranges):
            warnings.warn(
                "Voltage ranges for the new processor are different "
                "than the control voltage ranges for which the DNPU was trained: \n\n"
                f"* old voltage ranges: \n {old_ranges.cpu().numpy()} \n\n"
                f"* new voltage ranges: \n {new_ranges.cpu().numpy()} \n")
        del old_ranges
        del new_ranges

    # @TODO: Update documentation
    def sw_train(
        self,
        configs: dict,
        info: dict,
        model_state_dict: collections.OrderedDict = None,
    ):
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
        old_ranges = self.processor.get_voltage_ranges().clone().cpu().half()
        self.processor.swap(configs, info, model_state_dict)
        new_ranges = self.processor.get_voltage_ranges().clone().cpu().half()
        assert torch.equal(
            old_ranges,
            new_ranges), "Voltage ranges for the new processor are different "
        "than the control voltage ranges for which the DNPU was trained."
        del old_ranges
        del new_ranges

    def set_control_voltages(self, control_voltages: torch.Tensor):
        """
        Change the control_voltages of the network.

        Example
        -------
        >>> dnpu.set_control_voltages(torch.tensor([1.0, 2.0, 3.0, 4.0]))

        Parameters
        ----------
        control_voltages : torch.Tensor
            New value of the control_voltages.
            One dimensional tensor.
        """
        with torch.no_grad():
            assert (
                self.control_voltages.shape == control_voltages.shape
            ), "Control voltages could not be set due to a shape missmatch "
            "with regard to the ones already in the model."
            assert (
                self.control_voltages.dtype == control_voltages.dtype
            ), "Control voltages could not be set due to a shape missmatch "
            "with regard to the ones already in the model."
            self.control_voltages = torch.nn.Parameter(
                control_voltages.type_as(self.control_voltages).to(
                    self.control_voltages.device))

    def get_control_voltages(self) -> torch.Tensor:
        """
        Get the (next) control_voltages of the network, detach it from
        the computational graph.

        Returns
        -------
        torch.Tensor
            Value of the control_voltages.
        """
        return self.control_voltages.detach()

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
        Reset the control_voltages of the processor.
        """
        del self.control_voltages
        self._init_control_voltages()

    # TODO: Document the need to override the closing of the processor on
    # custom models.
    def close(self):
        """
        Close the processor. For simulation models, it does nothing. For hardware models it closes
        the drivers.
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
    input_data_indices: torch.Tensor,
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
    assert (
        input_data.dtype == control_data.dtype
        and input_data.device == control_data.device
    ), ("Input data voltages and control voltages have a different data type "
        + "or are in a different device (CUDA or CPU). ")
    result = torch.empty(
        (input_data.shape[0], len(input_data_indices) + len(control_indices)),
        device=input_data.device,
        dtype=input_data.dtype)
    result[:, input_data_indices] = input_data
    result[:, control_indices] = control_data
    return result
