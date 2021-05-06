import torch.nn as nn
import warnings
import collections

from brainspy.processors.simulation.processor import SurrogateModel
from brainspy.processors.hardware.processor import HardwareProcessor


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
        if not hasattr(self, "processor") or self._get_configs() != configs:
            if configs["processor_type"] == "simulation":
                self.processor = SurrogateModel(
                    info["model_structure"], model_state_dict
                )
                self.processor.set_effects_from_dict(
                    info["electrode_info"], configs["electrode_effects"]
                )
            elif (
                configs["processor_type"] == "cdaq_to_cdaq" or configs["processor_type"] == "cdaq_to_nidaq"
            ):
                configs["driver"]["instrument_type"] = configs["processor_type"]
                if "activation_voltages" not in configs["driver"]["instruments_setup"]:
                    configs["driver"]["instruments_setup"][
                        "activation_voltages"
                    ] = info["electrode_info"]["activation_electrodes"][
                        "voltage_ranges"
                    ]
                self.processor = HardwareProcessor(
                    configs["driver"],
                    configs["waveform"]["slope_length"],
                    configs["waveform"]["plateau_length"],
                )
                warnings.warn(
                    f"The hardware setup has been initialised with regard to a model trained with the following parameters. \nPlease make sure that the configurations of your hardware setup match these values: \n\t * An amplification correction of {self.info['electrode_info']['output_electrodes']['amplification']}\n\t * A clipping value range between {self.info['electrode_info']['output_electrodes']['clipping_value']}\n\t * Voltage ranges within {self.info['electrode_info']['activation_electrodes']['voltage_ranges']} "
                )
                if "amplification" in configs["driver"]:
                    warnings.warn(
                        f"The amplification has been overriden by the user to a value of: {configs['driver']['amplification']}"
                    )
                else:
                    configs["driver"]["amplification"] = self.info["electrode_info"][
                        "output_electrodes"
                    ]["amplification"]

            elif configs["processor_type"] == "simulation_debug":
                driver = SurrogateModel(info["model_structure"], model_state_dict)
                driver.set_effects_from_dict(
                    info["electrode_info"], configs["electrode_effects"]
                )
                self.processor = HardwareProcessor(driver_configs=driver)
            else:
                raise NotImplementedError(
                    f"Platform {configs['platform']} is not recognised. The platform has to be either simulation, simulation_debug, cdaq_to_cdaq or cdaq_to_nidaq. "
                )

    def forward(self, x):
        return self.processor(x)

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
            warnings.warn("Instance of processor not recognised.")
            return None

    def is_hardware(self):
        """
        Method to indicate whether this is a hardware processor. Returns
        False if it is a hardware setup and returns True if it is a simulation.

        Returns
        -------
        bool
        """
        return self.processor.is_hardware()

    def close(self):
        self.processor.close()