from typing import Sequence, Union
import torch
import torch.nn as nn
from torch import Tensor
import numpy as np

from brainspy.processors.simulation.processor import SurrogateModel
from brainspy.processors.hardware.processor import HardwareProcessor

from brainspy.utils.pytorch import TorchUtils


class Processor(nn.Module):
    # A class for handling the usage and swapping of hardware and software processors
    # It handles merging the data with control voltages as well as relevant information for the voltage ranges

    def __init__(self, arg):
        super(Processor, self).__init__()
        self.load_processor(arg)

    def load_processor(self, arg):
        # @TODO: Thoroughly document the different configurations available for each processor
        if isinstance(arg, dict):
            self._load_processor_from_configs(arg)
        elif isinstance(arg, SurrogateModel):
            self.processor = arg
        elif isinstance(arg, HardwareProcessor):
            self.processor = arg
        else:
            assert (
                False
            ), "The processor can either be a valid configuration dictionary, or an instance of either HardwareProcessor or SurrogateModel"
        self._init_electrode_info(arg["data"])
        self.is_hardware = self.processor.is_hardware()

    def _load_processor_from_configs(self, configs):
        if not hasattr(self, "processor") or self._get_configs() != configs:
            if configs["processor_type"] == "simulation":
                self.processor = SurrogateModel(configs["driver"]["torch_model_dict"])
                self.processor.set_effects_from_dict(configs["driver"])
            elif (
                configs["processor_type"] == "simulation_debug"
                or configs["processor_type"] == "cdaq_to_cdaq"
                or configs["processor_type"] == "cdaq_to_nidaq"
            ):
                self.processor = HardwareProcessor(configs)
            else:
                raise NotImplementedError(
                    f"Platform {configs['platform']} is not recognised. The platform has to be either simulation, simulation_debug, cdaq_to_cdaq or cdaq_to_nidaq. "
                )

    def _init_electrode_info(self, configs):
        self.electrode_no = self.processor.get_electrode_no()
        if self.electrode_no is not None:
            assert (
                self.electrode_no == configs["input_electrode_no"]
            ), "The input electrode number does not coincide with the one specified in the configs."
        else:
            self.electrode_no = configs["input_electrode_no"]

        # self.input_no = len(configs['data_input_indices'])
        self.data_input_indices = TorchUtils.format(
            configs["input_indices"], data_type=torch.int64
        )

        self.control_indices = np.delete(
            np.arange(configs["input_electrode_no"]), configs["input_indices"]
        )
        self.control_indices = TorchUtils.format(
            self.control_indices, data_type=torch.int64
        )  # IndexError: tensors used as indices must be long, byte or bool tensors

    def forward(self, data, control_voltages):
        merged_data = merge_electrode_data(
            data,
            control_voltages,
            self.data_input_indices,
            self.control_indices,
        )
        return self.processor(merged_data)

    def get_input_ranges(self):
        return self.processor.voltage_ranges[self.data_input_indices]

    def get_control_ranges(self):
        return self.processor.voltage_ranges[self.control_indices]

    def get_clipping_value(self):
        return self.processor.clipping_value

    def _get_configs(self):
        if isinstance(self.processor, HardwareProcessor):
            return self.processor.driver.configs
        elif isinstance(self.processor, SurrogateModel):
            return self.processor.configs
        else:
            print("Warning: Instance of processor not recognised.")
            return None

    def close(self):
        self.processor.close()


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
