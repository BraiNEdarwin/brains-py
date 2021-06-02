import torch

from brainspy.processors.processor import Processor
from brainspy.processors.dnpu import DNPU


class DNPUConv2d(DNPU):
    def __init__(
        self,
        processor,
        data_input_indices: list,  # Data input electrode indices. It should be a double list (e.g., [[1,2]] or [[1,2],[1,3]])
        in_channels: int,
        out_channels: int,
        kernel_size,  # TODO: put datatype as : _size_2_t format
        stride=1,
        padding=0,
        dilation=1,
        postprocess_type="sum",
        forward_pass_type: str = 'vec',
    ):
        super(DNPUConv2d, self).__init__(processor, data_input_indices, forward_pass_type=forward_pass_type)

        self.raw_inputs_list = data_input_indices # data_input_indices TO BE REMOVED!
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.unfold = torch.nn.Unfold(kernel_size=kernel_size, stride=stride, padding=padding)

        # @TODO: Are kernel size and stride needed as self. parameters?
        self.input_transform = False
        self.batch_norm = False

        if isinstance(processor, Processor):
            self.processor = processor
        else:
            self.processor = Processor(
                processor
            )  # It accepts initialising a processor as a dictionary

        # IndexError: tensors used as indices must be long, byte or bool tensors
        self.postprocess_type = postprocess_type
        if postprocess_type == "linear":
            self.linear_nodes = torch.nn.Linear(self.get_node_no(), 1, bias=False)
            #self.linear_kernels = torch.nn.Linear(self.out_channels, 1)
        self.init_params()

    def init_params(self):
        # -- Setup node --
        control_shape = list(self.control_indices.shape)
        control_shape.insert(0, self.out_channels)
        control_shape.insert(0, self.in_channels)

        self.control_indices = self.control_indices.expand(control_shape)

        control_shape.append(2)  # Extra dimension for minimum and maximum in control ranges
        self.control_ranges = self.control_ranges.expand(control_shape)

        # -- Set everything as torch Tensors and send to DEVICE --
        data_input_shape = list(self.data_input_indices.shape)
        data_input_shape.insert(0, self.out_channels)
        data_input_shape.insert(0, self.in_channels)

        self.data_input_indices = self.data_input_indices.expand(data_input_shape)

        self.reset()  # Apply a reset to the bias so that it gets initialised with the new adjustments to control indices

    def add_input_transform(self, input_range, strict=True):
        super(DNPUConv2d, self).add_input_transform(input_range, strict=strict)
        # Possible improvement, calculate the expected window size and expand it before the forward pass to have it pre-computed

    def add_batch_norm(
        self,
        eps=1e-05,
        momentum=0.1,
        affine=False,
        track_running_stats=True,
        clamp_at=None,
    ):
        self.batch_norm = True
        self.bn = torch.nn.BatchNorm3d(
            self.in_channels,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )
        self.clamp_at = clamp_at

    def remove_batch_norm(self):
        self.batch_norm = False
        del self.bn
        del self.clamp_at

    def get_output_dim(self, dim):
        return int(((dim + (2 * self.padding) - self.kernel_size) / self.stride) + 1)

    def preprocess(self, x):
        # Output from unfolding is [batch_size, window_size, window_no], where window_size = in_channel_no * img_width * img_height
        x = self.unfold(x)
        # Transpose the window_size dimension by the window_no dimension
        x = x.transpose(1, 2)

        # Reshape as: [Batch_size, window_no, in_chanels, node_no, input_electrode_no], where node_no is the number of DNPUs
        x = x.reshape(x.shape[0], x.shape[1], self.in_channels, self.get_node_no(), self.get_data_input_electrode_no())

        if self.batch_norm:
            x = self.apply_batch_norm(x)

        if self.input_transform:
            x = self.apply_input_transform(x)

        # Repeat info that will be used for each DNPU kernel
        # Shape as: [Batch_size, window_no, in_chanels, out_channels, node_no, input_electrode_no], where node_no is the number of DNPUs
        x = x.unsqueeze(3).expand(x.shape[0], x.shape[1], x.shape[2], self.out_channels, x.shape[3], x.shape[4])

        return x

    def apply_batch_norm(self, x):
        x = self.bn(x)
        if self.clamp_at is not None:
            x = x.clamp(-self.clamp_at, self.clamp_at)
        return x

    def apply_input_transform(self, x):
        scale = self.scale.expand_as(x)
        offset = self.offset.expand_as(x)
        x = (x * scale) + offset
        return x

    def merge_electrode_data(self, x):
        # Expand controls according to batch_size and window_no
        controls_shape = list(self.bias.shape)
        controls_shape.insert(0, x.shape[1])  # Add window_no dimension
        controls_shape.insert(0, x.shape[0])  # Add batch_size dimension
        controls = self.bias.expand(controls_shape)

        # Expand indices according to batch size
        control_indices = self.control_indices.expand(controls_shape)
        input_indices = self.data_input_indices.expand_as(x)
        original_data_dim = x.shape

        # Create input data and order it according to the indices
        last_dim = len(controls.shape) - 1  # For concatenating purposes
        indices = torch.argsort(torch.cat((input_indices, control_indices), dim=last_dim), dim=last_dim)

        data = torch.cat((x, controls), dim=last_dim)
        data = torch.gather(data, last_dim, indices)
        data = data.reshape(-1, data.shape[-1])

        return data, original_data_dim

    def postprocess(self, result, data_dim, output_dim):
        result = result.reshape(data_dim[:-1])
        result = result.sum(dim=2)  # Sum values from the input kernels

        if self.postprocess_type == "linear":
            # Pass the output from the DNPUs through a linear layer to combine them
            result = self.linear_nodes(result).squeeze(-1)
        elif self.postprocess_type == "sum":
            result = result.sum(dim=3)  # Sum the output from the devices used for the convolution

        result = result.transpose(1, 2)  # Return the output_kernel_no dimension to dimension 1.
        result = result.reshape(
            result.shape[0], result.shape[1], output_dim, -1
        )
        return result

    # Evaluate node
    def forward(self, x):
        output_dim = self.get_output_dim(x.shape[2])
        x = self.preprocess(x)
        x, original_data_dim = self.merge_electrode_data(x)
        x = self.processor(x)
        x = self.postprocess(x, original_data_dim, output_dim)

        return x
