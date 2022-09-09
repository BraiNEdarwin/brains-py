import torch

from brainspy.processors.processor import Processor
from brainspy.processors.dnpu import DNPU


class DNPUConv2d(DNPU):
    """
    A child of brainspy.processors.dnpu.DNPU class that performs a convolution operation with DNPUs.

    More information about the convolution operation can be found in:
    https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html?highlight=conv#torch.nn.Conv2d

    """
    def __init__(self,
                 processor,
                 data_input_indices: list,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 forward_pass_type: str = 'vec'):
        """
        Applies a conv2d operation with time multiplexing on a core DNPU processor.

        Attributes:
        processor : brainspy.processors.processor.Processor
            An instance of a Processor, which can hold a DNPU model or a driver connection to the
            DNPU hardware.

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

        in_channels : int
            Number of channels in the input image.

        out_channels : int
            Number of channels produced by the DNPU convolution.

        kernel_size : int or tuple
            Size of the convolving kernel.

        stride : Optional[int or tuple]
            Stride of the convolution. Default: 1

        padding : Optional[int, tuple or str]
            Number of pixels equal to zero added to all four sides of the input. Default: 0

        dilation : Optional[int or tuple]
            Spacing between kernel elements. Default: 1

        """
        super(DNPUConv2d, self).__init__(processor,
                                         data_input_indices,
                                         forward_pass_type=forward_pass_type)
        assert type(in_channels) is int, 'in_channels should be integer'
        assert type(out_channels) is int, 'out_channels should be integer'
        assert type(
            kernel_size
        ) is int, 'kernel_size should be integer. Only square kernel sizes are supported, represented by a single number.'
        assert type(stride) is int, 'in_channels should be integer'
        assert type(padding) is int, 'in_channels should be integer'
        assert (
            torch.tensor(data_input_indices).numel() == kernel_size**2
        ), "Data input indices should be defined as mapping a single kernel. E.g., for a 3x3 convolution you need 9 data input indices, represented as (dnpu_node_no=3, data_input_no_per_dnpu_node=3)."
        self.raw_inputs_list = data_input_indices  # data_input_indices TO BE REMOVED!
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.unfold = torch.nn.Unfold(kernel_size=kernel_size,
                                      stride=stride,
                                      padding=padding)

        # @TODO: Are kernel size and stride needed as self parameters?
        self.input_transform = False

        self.init_params()

    def init_params(self):
        """
        Initialises the control electrode indices and the data input electrode indices
        according to the size of the convolution. After that, reinitialises the control
        voltages (bias) that were initialised on the super call, but this time with the
        new dimensions for the control and data input indices.
        """
        # -- Setup node --
        control_shape = list(self.control_indices.shape)
        control_shape.insert(0, self.out_channels)
        control_shape.insert(0, self.in_channels)

        self.control_indices = self.control_indices.expand(
            control_shape).clone()

        control_shape.append(
            2)  # Extra dimension for minimum and maximum in control ranges
        self.control_ranges = self.control_ranges.expand(control_shape).clone()

        # -- Set everything as torch Tensors and send to DEVICE --
        data_input_shape = list(self.data_input_indices.shape)
        data_input_shape.insert(0, self.out_channels)
        data_input_shape.insert(0, self.in_channels)

        self.data_input_indices = self.data_input_indices.expand(
            data_input_shape).clone()
        # Apply a reset to the bias so that it gets initialised with the new adjustments
        # to control indices.
        self.reset()

    def add_input_transform(self, input_range, strict=True):
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
        """
        super(DNPUConv2d, self).add_input_transform(input_range, strict=strict)
        # Possible improvement, calculate the expected window size and expand it before
        # the forward pass to have it pre-computed

    def get_output_dim(self, dim):
        """
        Get the expected dimension of the output after the convolution.
        """
        # Tuple support has been dropped
        # if isinstance(self.stride, tuple):
        #     assert self.stride[0] == self.stride[
        #         1], "Different sized stride tuple not supported."
        #     stride = self.stride[0]
        # else:
        #     stride = self.stride

        return int(((dim +
                     (2 * self.padding) - self.kernel_size) / self.stride) + 1)

    def preprocess(self, x):
        """
        It extracts sliding local blocks from a batched input tensor. Then, it reshapes the
        input in a vectorised way, so that the input has the following a shape of
        (batch_size, dnpu_electrode_no). It applies batch norm and/or a linear transformation
        if these are added by calling add_input_transform after the
        initialisation of this module. These call only needs to happen once.

        Attributes:
        x : torch.Tensor
            The raw input data to the convolution.

        Returns
        -------
        torch.Tensor

        """
        # Output from unfolding is [batch_size, window_size, window_no],
        # where window_size = in_channel_no * img_width * img_height
        x = self.unfold(x)

        # Transpose the window_size dimension by the window_no dimension
        x = x.transpose(1, 2)

        # Reshape as: [Batch_size, window_no, in_chanels, node_no, input_electrode_no],
        # where node_no is the number of DNPUs
        x = x.reshape(x.shape[0], x.shape[1], self.in_channels,
                      self.get_node_no(), self.get_data_input_electrode_no())

        if self.input_transform:
            x = self._apply_input_transform(x)

        # Repeat info that will be used for each DNPU kernel
        # Shape as: [Batch_size, window_no, in_chanels, out_channels, node_no, input_electrode_no],
        # where node_no is the number of DNPUs.
        x = x.unsqueeze(3).expand(x.shape[0], x.shape[1], x.shape[2],
                                  self.out_channels, x.shape[3], x.shape[4])

        return x

    def _apply_input_transform(self, x):
        """
        Applies the input transformation before sending the data into the DNPU convolution.
        It is only applied if an external call to add_input_transform has been done after
        the initialisation of the module. It is applied after the data has been reshaped into
        [Batch_size, window_no, in_chanels, node_no, input_electrode_no].

        Attributes:
            x : torch.Tensor
            Input data, reshaped as [Batch_size, window_no, in_chanels, node_no, input_electrode_no]

        Returns
        -------
        x: torch.Tensor
            The input data after a linear transformation. The maximum and minimum ranges will be
            those of the maximum and minimum ranges of the data per electrode. The control
            electrodes are selected according to the data_input_indices attribute in the __init__
            method of the module. The maximumn and minimum voltage ranges are defined by the
            training data of the surrogate model.
        """
        if self.unique_transform:
            x = (x * self.scale) + self.offset
        else:
            scale = self.scale.expand_as(x)
            offset = self.offset.expand_as(x)
            x = (x * scale) + offset
        return x

    def merge_electrode_data(self, x):
        """
        Merge the input data to be fed to the input data electrodes with the
        data to be fed to the control voltage electrodes.

        Parameters
        ----------
        x: torch.tensor
            Input data that will be fed into the input data electrodes.

        Returns
        -------
        data: torch.Tensor
            A tensor with the input data and control voltage data to be fed
            through the activation electrodes, ordered according to the configurations
            of the indices for the data input and control voltage inputs to the
            DNPU convolution architecture. The data is given with a shape of:
            (batch_size,electrode_no).

        original_data_dim: torch.Size
            The original data dimensions of the data before being converted into a
            shape of (batch_size,electrode_no). This information is used to reconstruct
            the output tensor after is passed through the processor.  

        """
        # Expand controls according to batch_size and window_no
        controls_shape = list(self.control_voltages.shape)
        controls_shape.insert(0, x.shape[1])  # Add window_no dimension
        controls_shape.insert(0, x.shape[0])  # Add batch_size dimension
        controls = self.control_voltages.expand(controls_shape)

        # Expand indices according to batch size
        control_indices = self.control_indices.expand(controls_shape)
        input_indices = self.data_input_indices.expand_as(x)
        original_data_dim = x.shape

        # Create input data and order it according to the indices
        last_dim = len(controls.shape) - 1  # For concatenating purposes
        indices = torch.argsort(torch.cat((input_indices, control_indices),
                                          dim=last_dim),
                                dim=last_dim)

        data = torch.cat((x, controls), dim=last_dim)
        data = torch.gather(data, last_dim, indices)
        data = data.reshape(-1, data.shape[-1])

        return data, original_data_dim

    def postprocess(self, result, data_dim, output_dim):
        """
        The shape of the output of the convolution after passing through the processor is of
        (batch_size,electrode_no). This method does the final operations of the convolution,
        and to make the output have the same data shape as it would from outside a covolution.

        The postprocessing sums the values from the input kernel dimensions, and then applies
        either a sum or a linear operation to combine the outputs of the DNPU Convolution module.

        Parameters
        ----------
        result: torch.Tensor
            A tensor with the input data and control voltage data to be fed
            through the activation electrodes, ordered according to the configurations
            of the indices for the data input and control voltage inputs to the
            DNPU convolution architecture. The data is given with a shape of:
            (batch_size,electrode_no).

        data_dim: torch.Shape
            The original data dimensions of the data before being converted into a
            shape of (batch_size,electrode_no). This information is used to reconstruct
            the output tensor after is passed through the processor.

        output_dim: int
            Dimension of the output after the convolution. It can be calculated with the method
            get_output_dim of this module.

        Returns
        -------
        result: torch.Tensor
            Image out of the convolution. With a shape of (batch_size, channel_no,
            output_feature_height, output_feature_width)
        """
        result = result.reshape(data_dim[:-1])
        result = result.sum(dim=2)  # Sum values from the input kernels

        result = result.sum(
            dim=3)  # Sum the output from the devices used for the convolution

        result = result.transpose(
            1, 2)  # Return the output_kernel_no dimension to dimension 1.
        result = result.reshape(result.shape[0], result.shape[1], output_dim,
                                -1)
        return result

    # Evaluate node
    def forward(self, x):
        """
        Forward pass of the convolution module.

        Parameters
        ----------
        x: torch.Tensor
            Input to the convolution, with a shape of
            (batch_size, channel_no, input_img_height, input_image_width)

        Returns
        -------
        result: torch.Tensor
            Image out of the convolution. With a shape of (batch_size, channel_no,
            output_feature_height, output_feature_width)
        """
        output_dim = self.get_output_dim(x.shape[2])
        x = self.preprocess(x)
        x, original_data_dim = self.merge_electrode_data(x)
        x = self.processor(x)
        x = self.postprocess(x, original_data_dim, output_dim)

        return x
