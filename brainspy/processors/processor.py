import torch
import torch.nn as nn
import numpy as np

from brainspy.processors.simulation.processor import SurrogateModel
from brainspy.processors.hardware.processor import HardwareProcessor

from brainspy.utils.pytorch import TorchUtils
from brainspy.utils.electrodes import merge_electrode_data


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
                self.processor = SurrogateModel(configs["driver"])
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
        self.data_input_indices = TorchUtils.get_tensor_from_list(
            configs["input_indices"], data_type=torch.int64
        )

        self.control_indices = np.delete(
            np.arange(configs["input_electrode_no"]), configs["input_indices"]
        )
        self.control_indices = TorchUtils.get_tensor_from_list(
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
