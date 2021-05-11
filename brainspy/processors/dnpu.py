import torch
import collections
import numpy as np

from torch import nn, Tensor
from typing import Sequence, Union

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
        merged_data = merge_electrode_data(
            x,
            self.bias.expand(x.size()[0], -1),
            self.data_input_indices,
            self.control_indices,
        )
        return self.processor(merged_data)

    def regularizer(self):
        return torch.sum(
            torch.relu(self.control_low - self.bias) + torch.relu(self.bias - self.control_high)
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

    def reset(self):
        del self.bias
        self._init_bias()

    # TODO: Document the need to override the closing of the processor on custom models.
    def close(self):
        self.processor.close()

    def is_hardware(self):
        return self.processor.is_hardware()

    # TODO: Document the need to override the closing of the return of the info dictionary.
    def get_info_dict(self):
        return self.info


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
        (inputs.shape[0], len(input_indices) + len(control_voltage_indices))
    )
    if use_torch:
        result = TorchUtils.format(result)
    result[:, input_indices] = inputs
    result[:, control_voltage_indices] = control_voltages
    return result
