import torch
import torch.nn as nn
from torch import Tensor
from typing import Sequence, Union
import numpy as np
import collections

from brainspy.processors.simulation.processor import SurrogateModel
from brainspy.processors.hardware.processor import HardwareProcessor

from brainspy.utils.pytorch import TorchUtils
from brainspy.utils.electrodes import set_effects_from_dict


class Processor(nn.Module):
    # A class for handling the usage and swapping of hardware and software processors
    # It handles merging the data with control voltages as well as relevant information for the voltage ranges
    """[summary]

    Parameters
    ----------
    configs : dict
        processor_type
        input_indices
        [electrode_effects]
        [waveform]
        [driver]
    """

    def __init__(
        self,
        configs: dict,
        info: dict,
        model_state_dict: collections.OrderedDict = None,
    ):
        super(Processor, self).__init__()
        self.info = info
        self.load_processor(configs, info, model_state_dict)

    def load_processor(
        self, configs, info: dict, model_state_dict: collections.OrderedDict = None
    ):
        # @TODO: Thoroughly document the different configurations available for each processor
        self._load_processor_from_configs(configs, info, model_state_dict)
        # self._init_electrode_info(configs["input_indices"])
        self.is_hardware = self.processor.is_hardware()

    def _load_processor_from_configs(
        self,
        configs,
        info,
        model_state_dict: collections.OrderedDict = None,
    ):
        if not hasattr(self, "processor") or self._get_configs() != configs:
            if configs["processor_type"] == "simulation":
                processor = SurrogateModel(info["model_structure"], model_state_dict)
            elif (
                configs["processor_type"] == "cdaq_to_cdaq"
                or configs["processor_type"] == "cdaq_to_nidaq"
            ):
                configs["driver"]["processor_type"] = configs["processor_type"]
                processor = HardwareProcessor(configs["driver"], configs["waveform"])
            elif configs["processor_type"] == "simulation_debug":
                driver = SurrogateModel(info["model_structure"], model_state_dict)
                driver = set_effects_from_dict(
                    driver, info["electrode_info"], configs["electrode_effects"]
                )
                processor = HardwareProcessor(
                    configs["driver"], configs["waveform"], debug_driver=driver
                )
            else:
                raise NotImplementedError(
                    f"Platform {configs['platform']} is not recognised. The platform has to be either simulation, simulation_debug, cdaq_to_cdaq or cdaq_to_nidaq. "
                )
            self.processor = set_effects_from_dict(
                processor, info["electrode_info"], configs["electrode_effects"]
            )

    def forward(self, data, control_voltages):
        merged_data = merge_electrode_data(
            data,
            control_voltages,
            self.data_input_indices,
            self.control_indices,
        )
        return self.processor(merged_data)

    def get_voltage_ranges(self):
        return self.processor.voltage_ranges

    def get_activation_electrode_no(self):
        """
        Get the number of activation electrodes of the processor.

        Returns
        -------
        int
            The number of electrodes of the processor.
        """
        return self.info["electrode_info"]["activation_electrodes"]["electrode_no"]

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


if __name__ == "__main__":
    import torch

    NODE_CONFIGS = {}
    NODE_CONFIGS["processor_type"] = "simulation_debug"
    NODE_CONFIGS["input_indices"] = [2, 3]
    NODE_CONFIGS["electrode_effects"] = {}
    NODE_CONFIGS["electrode_effects"]["amplification"] = 3
    NODE_CONFIGS["electrode_effects"]["clipping_value"] = [-300, 300]
    # NODE_CONFIGS["electrode_effects"]["control_voltages"]
    NODE_CONFIGS["electrode_effects"]["noise"] = {}
    NODE_CONFIGS["electrode_effects"]["noise"]["noise_type"] = "gaussian"
    NODE_CONFIGS["electrode_effects"]["noise"]["variance"] = 0.6533523201942444
    NODE_CONFIGS["driver"] = {}
    NODE_CONFIGS["waveform"] = {}
    NODE_CONFIGS["waveform"]["plateau_length"] = 1
    NODE_CONFIGS["waveform"]["slope_length"] = 0

    model_dir = "/home/unai/Documents/3-Programming/bspy/smg/tmp/output/new_test_model/training_data_2021_04_22_105203/training_data.pt"
    model_data = torch.load(model_dir)

    sm = Processor(
        NODE_CONFIGS,
        model_data["info"],
        model_data["model_state_dict"],
    )

    sm2 = Processor(NODE_CONFIGS, model_data["info"])