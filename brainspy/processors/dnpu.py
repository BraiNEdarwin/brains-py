import torch
import numpy as np
from torch import nn

import collections
from brainspy.utils.pytorch import TorchUtils
from brainspy.processors.processor import Processor


class DNPU(nn.Module):
    """"""

    def __init__(
        self,
        configs: dict,  # Dictionary or processor
        info: dict = None,
        model_state_dict: collections.OrderedDict = None,
        processor: Processor = None,
    ):
        super(DNPU, self).__init__()
        assert (
            processor is not None or info is not None
        ), "The DNPU must be initialised either with a processor or an info dictionary"
        if processor is not None:
            self.processor = processor
        else:
            self.processor = Processor(configs, info, model_state_dict)
        self._init_electrode_info(configs["input_indices"])
        self._init_dnpu()

    def _init_dnpu(self):
        for (
            params
        ) in (
            self.parameters()
        ):  # Freeze parameters of the neural network of the surrogate model
            params.requires_grad = False
        self._init_bias()

    def _init_bias(self):
        self.control_low = self.get_control_ranges()[:, 0]
        self.control_high = self.get_control_ranges()[:, 1]
        assert any(
            self.control_low < 0
        ), "Min. Voltage is assumed to be negative, but value is positive!"
        assert any(
            self.control_high > 0
        ), "Max. Voltage is assumed to be positive, but value is negative!"
        bias = self.control_low + (self.control_high - self.control_low) * torch.rand(
            1,
            len(self.control_indices),
            dtype=torch.get_default_dtype(),
            device=TorchUtils.get_device(),
        )

        self.bias = nn.Parameter(bias)

    def _init_electrode_info(self, input_indices):

        # self.input_no = len(configs['data_input_indices'])
        self.data_input_indices = TorchUtils.format(
            input_indices, data_type=torch.int64
        )

        self.control_indices = np.delete(
            np.arange(self.processor.get_activation_electrode_no()), input_indices
        )
        self.control_indices = TorchUtils.format(
            self.control_indices, data_type=torch.int64
        )  # IndexError: tensors used as indices must be long, byte or bool tensors

    def forward(self, x):
        return self.processor(x, self.bias.expand(x.size()[0], -1))

    def regularizer(self):
        return torch.sum(
            torch.relu(self.control_low - self.bias)
            + torch.relu(self.bias - self.control_high)
        )

    def hw_eval(self, arg):
        self.eval()
        if isinstance(arg, Processor):
            self.processor = arg
        else:
            self.processor.load_processor(arg)
        assert torch.equal(
            self.control_low.cpu().half(),
            self.processor.get_control_ranges()[:, 0].cpu().half(),
        ), "Low control voltage ranges for the new processor are different than the control voltage ranges for which the DNPU was trained."
        assert torch.equal(
            self.control_high.cpu().half(),
            self.processor.get_control_ranges()[:, 1].cpu().half(),
        ), "High control voltage ranges for the new processor are different than the control voltage ranges for which the DNPU was trained."
        # self._init_electrode_info(hw_processor_configs)

    def set_control_voltages(self, bias):
        with torch.no_grad():
            bias = bias.unsqueeze(dim=0)
            assert (
                self.bias.shape == bias.shape
            ), "Control voltages could not be set due to a shape missmatch with regard to the ones already in the model."
            self.bias = torch.nn.Parameter(TorchUtils.format(bias))

    def get_control_voltages(self):
        return next(self.parameters()).detach()

    def get_input_ranges(self):
        return self.processor.get_voltage_ranges()[self.data_input_indices]

    def get_control_ranges(self):
        return self.processor.get_voltage_ranges()[self.control_indices]

    def get_clipping_value(self):
        return self.processor.get_clipping_value()

    # TODO: Check if this function is really needed or if it needs to be completed.
    def reset(self):
        for k in range(len(self.control_low)):
            # print(f'    resetting control {k} between : {self.control_low[k], self.control_high[k]}')
            self.bias.data[:, k].uniform_(self.control_low[k], self.control_high[k])

    # TODO: Document the need to override the closing of the processor on custom models.
    def close(self):
        self.processor.close()

    def is_hardware(self):
        return self.processor.is_hardware

    # TODO: Document the need to override the closing of the return of the info dictionary.
    def get_info_dict(self):
        return self.info
