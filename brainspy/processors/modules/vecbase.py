import torch
import numpy as np
import torch.nn as nn
from brainspy.utils.pytorch import TorchUtils
from brainspy.processors.processor import Processor

from brainspy.utils.transforms import get_linear_transform_constants


class DNPUBase(nn.Module):
    def __init__(self, processor, inputs_list):
        super(DNPUBase, self).__init__()
        if isinstance(processor, Processor):
            self.processor = processor
        else:
            self.processor = Processor(
                processor
            )  # It accepts initialising a processor as a dictionary
        self.device_no = len(inputs_list)
        # ######## Set up node #########
        # Freeze parameters of node
        for params in self.processor.parameters():
            params.requires_grad = False

        self.indices_node = np.arange(
            len(self.processor.data_input_indices) + len(self.processor.control_indices)
        )
        # ######## set learnable parameters #########
        self.control_list = TorchUtils.format(
            self.set_controls(inputs_list), data_type=torch.int64
        )

        # ######## Initialise data input ranges #########
        self.data_input_low = torch.stack(
            [
                self.processor.processor.voltage_ranges[indx_cv, 0]
                for indx_cv in inputs_list
            ]
        )
        self.data_input_high = torch.stack(
            [
                self.processor.processor.voltage_ranges[indx_cv, 1]
                for indx_cv in inputs_list
            ]
        )

        # ##### Set everything as torch Tensors and send to DEVICE ######
        self.inputs_list = TorchUtils.format(inputs_list, data_type=torch.int64)
        # IndexError: tensors used as indices must be long, byte or bool tensors

    def set_controls(self, inputs_list):
        control_list = [np.delete(self.indices_node, indx) for indx in inputs_list]
        self.control_low = torch.stack(
            [
                self.processor.processor.voltage_ranges[indx_cv, 0]
                for indx_cv in control_list
            ]
        )
        self.control_high = torch.stack(
            [
                self.processor.processor.voltage_ranges[indx_cv, 1]
                for indx_cv in control_list
            ]
        )

        # Sample control parameters
        # controls = [
        #     self.sample_controls(low, high)
        #     for low, high in zip(control_low, control_high)
        # ]

        # Register as learnable parameters
        # self.all_controls = nn.Parameter(torch.stack(controls).squeeze())
        self.all_controls = nn.Parameter(self.sample_controls(len(control_list[0])))

        return control_list

    # def sample_controls(self, low, high):
    #     samples = torch.rand(1, len(low), device=TorchUtils.get_device(), dtype=torch.get_default_dtype())
    #     return low + (high - low) * samples

    def sample_controls(self, control_no):
        # @TODO: This code is very similar to that of add_transform. Create an additional function to avoid repeating code.
        data_input_range = [0, 1]
        output_range = self.get_control_ranges()  # .flatten(1,-1)
        min_input = data_input_range[0]
        max_input = data_input_range[1]
        input_range = torch.ones_like(output_range)
        input_range[0] *= min_input
        input_range[1] *= max_input
        amplitude, offset = get_linear_transform_constants(
            output_range[0], output_range[1], input_range[0], input_range[1]
        )
        samples = torch.rand(
            (self.device_no, control_no),
            device=TorchUtils.get_device(),
            dtype=torch.get_default_dtype(),
        )
        return (amplitude * samples) + offset

    # Evaluate node

    def forward(self, x):
        batch_size = x.size(0)

        # Reshape input and expand controls
        x = x.reshape(batch_size, self.inputs_list.shape[0], self.inputs_list.shape[1])
        last_dim = len(x.size()) - 1
        controls = self.all_controls.unsqueeze(0).repeat_interleave(batch_size, dim=0)

        # If necessary apply a transformation to the input ranges
        if not (self.amplitude is None):
            amplitude = self.amplitude.unsqueeze(0).repeat_interleave(batch_size, dim=0)
            offset = self.offset.unsqueeze(0).repeat_interleave(batch_size, dim=0)
            x = (x * amplitude) + offset

        # Expand indices according to batch size
        input_indices = self.inputs_list.unsqueeze(0).repeat_interleave(
            batch_size, dim=0
        )
        control_indices = self.control_list.unsqueeze(0).repeat_interleave(
            batch_size, dim=0
        )

        # Create input data and order it according to the indices
        indices = torch.cat((input_indices, control_indices), dim=last_dim)
        data = torch.cat((x, controls), dim=last_dim)
        data = torch.gather(data, last_dim, indices)

        # pass data through the processor
        return self.processor.processor(data).squeeze()  # * self.node.amplification

    def add_transform(self, data_input_range, clip_input=False):
        output_range = self.get_input_ranges()  # .flatten(1,-1)
        min_input = data_input_range[0]
        max_input = data_input_range[1]
        input_range = torch.ones_like(output_range)
        input_range[0] *= min_input
        input_range[1] *= max_input

        self.amplitude, self.offset = get_linear_transform_constants(
            output_range[0], output_range[1], input_range[0], input_range[1]
        )
        # self.VariableRangeMapper()
        # self.transform = SimpleMapping(input_range=[-0.4242,2.8215], output_range=self.get_input_ranges().flatten(1,-1), clip_input=clip_input)

    def reset(self):
        raise NotImplementedError("Resetting controls not implemented!!")
        # for k in range(len(self.control_low)):
        #     # print(f'    resetting control {k} between : {self.control_low[k], self.control_high[k]}')
        #     self.controls.data[:, k].uniform_(self.control_low[k], self.control_high[k])

    def regularizer(self):
        if "control_low" in dir(self) and "control_high" in dir(self):
            return 0
        else:
            assert any(
                self.control_low.min(dim=0)[0] < 0
            ), "Min. Voltage is assumed to be negative, but value is positive!"
            assert any(
                self.control_high.max(dim=0)[0] > 0
            ), "Max. Voltage is assumed to be positive, but value is negative!"
            buff = 0.0
            for i, p in enumerate(self.all_controls):
                buff += torch.sum(
                    torch.relu(self.control_low[i] - p) + torch.relu(p - self.control_high[i])
                )
            return buff

    def hw_eval(self, hw_processor_configs):
        self.processor.hw_eval(hw_processor_configs)

    def is_hardware(self):
        return self.processor.is_hardware

    def get_clipping_value(self):
        return self.processor.get_clipping_value()

    def get_input_ranges(self):
        return torch.stack((self.data_input_low, self.data_input_high))

    def get_control_ranges(self):
        return torch.stack(
            (self.control_low, self.control_high)
        )  # Total Dimensions 3: Dim 0: 0=min volt range1=max volt range, Dim 1: Index of node, Dim 2: Index of electrode

    def get_control_voltages(self):
        return (
            self.all_controls.detach()
        )  # torch.vstack([cv.data.detach() for cv in self.all_controls]).flatten()

    def set_control_voltages(self, control_voltages):
        with torch.no_grad():
            # bias = bias.unsqueeze(dim=0)
            assert (
                self.all_controls.shape == control_voltages.shape
            ), "Control voltages could not be set due to a shape missmatch with regard to the ones already in the model."
            self.all_controls = torch.nn.Parameter(TorchUtils.format(control_voltages))
