"""
Contains the main processor class. Creates either a simulation processor or a
hardware processor with given settings.
Also handles merging the data with control voltages.
"""

import warnings
import collections
from typing import Union

import torch
import torch.nn as nn

from brainspy.utils.waveform import WaveformManager
from brainspy.processors.simulation.processor import SurrogateModel
from brainspy.processors.hardware.processor import HardwareProcessor


class Processor(nn.Module):
    """
    A class for handling the usage and swapping of hardware and software
    processors.

    Attributes
    ----------
    configs : dict
        Configs dictionary, documented in init method.
    info : dict
        Info dictionary, documented in init method.
    model_state_dict : collections.OrderedDict, optional. Documented in init method.
    """
    def __init__(
        self,
        configs: dict,
        info: dict,
        model_state_dict: collections.OrderedDict = None,
    ):
        """
        Create a processor and run load_processor, which creates either a
        simulation processor or a hardware processor from the given
        dictionaries.

        Parameters
        ----------
        configs : dict
            processor_type : str
                Type of processor, can be
                "simulation",
                "cdaq_to_cdaq" (hardware),
                "cdaq_to_nidaq" (hardware), or
                "simulation_debug".
            electrode_effects : dict
                (Only for simulation)
                Electode effects for simulation processors.
                Documented in SurrogateModel, set effects method.
            waveform:
                slope_length : int
                    Length of the slopes, see waveform.py.
                plateau_length : int
                    Length of the plateaus, see waveform.py.
            driver:
                Only for hardware, refer to HardwareProcessor for required keys.
        info : dict
            model_structure : dict
                Dimensions of the neural network.
                Documented in SurrogateModel, init method.
            electrode_info : dict
                Dictionary containing the default values for the electrodes in
                the simulation processor.
                Documented in SurrogateModel, set effects method (see info).
        model_state_dict : collections.OrderedDict, optional
            State dictionary of the simulation model,
            parameters of the pytorch neural network; if not given, will be
            initialized with random parameters (weights and biases);
            by default None.
        """
        super(Processor, self).__init__()
        self.load_processor(configs, info, model_state_dict)
        self.waveform_mgr = WaveformManager(configs['waveform'])

    def load_processor(self,
                       configs: dict,
                       info: dict,
                       model_state_dict: collections.OrderedDict = None):
        """
        Create a processor depending on the provided settings.

        Info dictionary will override existing one if not None, otherwise
        existing one will be used.

        For simulation, simply create the processor and set the effects (as is
        required).

        For hardware, first check if the activation voltage ranges are in
        configs. If not, take them from electrode_info in the info dictionary.
        Create a warning with the values of the electrode effects. Make sure
        the amplification is set.

        Method is called by constructor but can also be called externally.

        Parameters
        ----------
        configs : dict
            Configs dictionary, documented in init method.
        info : dict, optional
            Info dictionary, documented in init method, by default None.
        model_state_dict : collections.OrderedDict, optional
            State dictionary for the simulation model, by default None.
            Documented in init method.

        Raises
        ------
        NotImplementedError
            In case the processor type is not recognized.
        UserWarning
            When a hardware processor is created; contains the values of the
            electrode effects.
        """

        self.processor: Union[SurrogateModel, HardwareProcessor]
        self.info = info
        self.configs = configs

        # create SurrogateModel
        if configs["processor_type"] == "simulation":
            self.processor = SurrogateModel(
                model_structure=self.info["model_structure"],
                model_state_dict=model_state_dict)
            self.processor.set_effects_from_dict(
                info=self.info["electrode_info"],
                configs=configs["electrode_effects"])

        # create hardware processor
        elif (configs["processor_type"] == "cdaq_to_cdaq"
              or configs["processor_type"] == "cdaq_to_nidaq"):

            # set instrument type
            configs["driver"]["instrument_type"] = configs["processor_type"]

            # check if activation voltage ranges is in configs;
            # if not, take it from electrode info
            if "activation_voltage_ranges" not in configs["driver"][
                    "instruments_setup"]:
                configs["driver"]["instruments_setup"][
                    "activation_voltage_ranges"] = self.info["electrode_info"][
                        "activation_electrodes"]["voltage_ranges"]

            # create warning with electrode info
            electrode_info = self.info["electrode_info"]  # avoid line too long
            warnings.warn(
                "The hardware setup has been initialised with regard to a "
                "model trained with the following parameters.\n"
                "Please make sure that the configurations of your hardware "
                "setup match these values:\n"
                "\t * An amplification correction of "
                f"{electrode_info['output_electrodes']['amplification']}\n"
                "\t * A clipping value range between "
                f"{electrode_info['output_electrodes']['clipping_value']}\n"
                "\t * Voltage ranges within "
                f"{electrode_info['activation_electrodes']['voltage_ranges']}")

            # make sure amplification is set and raise warning if it is;
            # otherwise take it from electrode_info
            if "amplification" in configs["driver"]:
                warnings.warn(
                    "The amplification has been overriden by the user to a "
                    f"value of: {configs['driver']['amplification']}")
            else:
                configs["driver"]["amplification"] = self.info[
                    "electrode_info"]["output_electrodes"]["amplification"]

            self.processor = HardwareProcessor(
                configs["driver"],
                configs["waveform"]["slope_length"],
                configs["waveform"]["plateau_length"],
            )

        # create simulation processor for debugging
        elif configs["processor_type"] == "simulation_debug":
            driver = SurrogateModel(self.info["model_structure"],
                                    model_state_dict)
            driver.set_effects_from_dict(self.info["electrode_info"],
                                         configs["electrode_effects"])
            self.processor = HardwareProcessor(
                instrument_configs=driver,
                slope_length=configs["waveform"]["slope_length"],
                plateau_length=configs["waveform"]["plateau_length"])

        # processor type not recognized
        else:
            raise NotImplementedError(
                f"Platform {configs['processor_type']} is not recognized. The "
                "platform has to be either simulation, simulation_debug, "
                "cdaq_to_cdaq or cdaq_to_nidaq. ")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run a forward pass through the processor. It creates plateaus from data points before
        sending the data to the simulation or hardware processor. The hardware processor will
        internally create the slopes to the plateaus. The simulation processor does not need slopes.

        Example
        -------
        >>> x = torch.tensor([1, 2, 3, 4])
        >>> self.forward(x)
        torch.Tensor([1])

        Forward pass with one data point (4 activation electrodes).

        Parameters
        ----------
        x : torch.Tensor
            Input data. It is expected to have a shape of [batch_size, activation_electrode_no].

        Returns
        -------
        torch.Tensor
            Output data.
        """
        x = self.waveform_mgr.points_to_plateaus(x)
        return self.processor(x)

    def format_targets(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transform data to plateaus so that target data can be compared to
        output data.

        Example
        -------
        >>> x = torch.tensor([[1], [2]])
        >>> self.format_targets(x)
        torch.Tensor([[1], [1], [1], [2], [2], [2]])

        Point data is transformed to plateau data where the plateau length
        is 3.

        Parameters
        ----------
        x : torch.Tensor
            Data to be transformed.

        Returns
        -------
        torch.Tensor
            Transformed data.
        """
        return self.waveform_mgr.points_to_plateaus(x)

    def get_voltage_ranges(self) -> torch.Tensor:
        """
        Get the voltage ranges of input of the the processor,
        whether it is hardware or simulation.

        Returns
        -------
        torch.Tensor
            Voltage ranges.
        """
        return self.processor.voltage_ranges

    def get_activation_electrode_no(self):
        """
        Get the number of activation electrodes of the processor.

        Returns
        -------
        int
            The number of activation electrodes of the processor.
        """
        return self.info["electrode_info"]["activation_electrodes"][
            "electrode_no"]

    def get_readout_electrode_no(self):
        """
        Get the number of readout electrodes of the processor.

        Returns
        -------
        int
            The number of readout electrodes of the processor.
        """
        return self.info["electrode_info"]["output_electrodes"]["electrode_no"]

    def get_clipping_value(self):
        """
        Get the output clipping. For hardware, take it from the info
        dictionary. For simulation, use the existing method.

        Returns
        -------
        torch.Tensor or list
            The output clipping of the processor.
        """
        if self.processor.is_hardware():
            return self.info['electrode_info']['output_electrodes'][
                'clipping_value']  # will return list
        else:
            return self.processor.get_clipping_value()  # will return tensor

    def swap(self,
             configs: dict,
             info: dict,
             model_state_dict: collections.OrderedDict = None):
        """
        Re-initialize: load the processor again from the given dictionaries.

        Parameters
        ----------
        configs : dict
            Configs dictionary, documented in init method of this class.
        info : dict
            Info dictionary, documented in init method of this class.
        model_state_dict : collections.OrderedDict, optional
            State dictionary for the simulation model, by default None.
            Documented in init method of this class.
        """
        self.load_processor(configs, info, model_state_dict)
        self.waveform_mgr = WaveformManager(configs['waveform'])

    def is_hardware(self) -> bool:
        """
        Method to indicate whether this is a hardware processor.

        Returns
        -------
        bool
            True if hardware, False if software.
        """
        return self.processor.is_hardware()

    def close(self):
        """
        Closes the driver related to the NI Tasks if the main processor is
        hardware.
        If the main processor is a simulation model, this does
        nothing.
        """
        self.processor.close()
