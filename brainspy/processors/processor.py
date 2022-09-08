"""
Contains the main processor class. Creates either a simulation processor or a
hardware processor with given settings.
Also handles merging the data with control voltages.
"""

import warnings
import collections
from typing import Union

import torch
import copy
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
    model_state_dict : collections.OrderedDict, optional. Documented in
    init method.
    average_plateaus: bool
        When there are plateaus that are higher than 1, wether if average the whole plateau or not.
    """
    def __init__(self,
                 configs: dict,
                 info: dict = None,
                 model_state_dict: collections.OrderedDict = None,
                 average_plateaus: bool = True):
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
                Only for hardware, refer to HardwareProcessor for keys.
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
        average_plateaus: bool
            When there are plateaus that are higher than 1, wether if average the whole plateau or not.
        """
        super(Processor, self).__init__()
        self.load_processor(configs, info, model_state_dict)
        self.waveform_mgr = WaveformManager(configs['waveform'])
        self.average_plateaus = average_plateaus

    def load_processor(self,
                       configs: dict,
                       info: dict = None,
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
        electrode_info_loaded = False

        # create SurrogateModel
        if configs["processor_type"] == "simulation":
            assert self.info is not None, "Simulation processor requires to be provided with an info dictionary in the constructor."
            self.processor = SurrogateModel(
                model_structure=self.info["model_structure"],
                model_state_dict=model_state_dict)
            if 'electrode_effects' in configs:
                self.processor.set_effects_from_dict(
                    info=self.info["electrode_info"],
                    configs=configs["electrode_effects"])
            else:
                self.processor.set_effects_from_dict(
                    info=self.info["electrode_info"])

        # create hardware processor
        elif (configs["processor_type"] == "cdaq_to_cdaq"
              or configs["processor_type"] == "cdaq_to_nidaq"):

            # set instrument type
            configs["driver"]["instrument_type"] = configs["processor_type"]
            if self.info is None:
                self.info = {}
                self.info['electrode_info'] = get_electrode_info(configs)
                electrode_info_loaded = True
            else:
                # create overwriting warning with electrode info
                electrode_info = self.info[
                    "electrode_info"]  # avoid line too long
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
                    f"{electrode_info['activation_electrodes']['voltage_ranges']}"
                )

            # check if activation voltage ranges is in configs;
            # if not, take it from electrode info
            if "activation_voltage_ranges" not in configs["driver"][
                    "instruments_setup"]:
                configs["driver"]["instruments_setup"][
                    "activation_voltage_ranges"] = self.info["electrode_info"][
                        "activation_electrodes"]["voltage_ranges"]

            # make sure amplification is set and raise warning if it is;
            # otherwise take it from electrode_info
            if "amplification" in configs[
                    "driver"] and not electrode_info_loaded:
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
            assert self.info is not None, "Simulation_debug processor requires to be provided with an info dictionary in the constructor."
            driver = SurrogateModel(self.info["model_structure"],
                                    model_state_dict)
            if 'electrode_effects' in configs:
                driver.set_effects_from_dict(self.info["electrode_info"],
                                             configs["electrode_effects"])
            else:
                driver.set_effects_from_dict(self.info["electrode_info"])
            driver.configs = copy.deepcopy(configs)
            if 'instruments_setup' not in driver.configs:
                driver.configs['instruments_setup'] = {}
                driver.configs['instruments_setup']['activation_channels'] = [
                    1
                ] * self.info['electrode_info']['activation_electrodes'][
                    'electrode_no']
            #[
            #     'activation_electrodes'] = self.info['electrode_info'][
            #         'activation_electrodes']

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

        Parameters
        ----------
        x : torch.Tensor
            Input data. It is expected to have a shape of [batch_size, activation_electrode_no].

        Returns
        -------
        torch.Tensor
            Output data.
        """
        if not (self.waveform_mgr.plateau_length == 1
                and self.waveform_mgr.slope_length == 0):
            x = self.waveform_mgr.points_to_plateaus(x)
        x = self.processor(x)
        if not (self.waveform_mgr.plateau_length == 1 and
                self.waveform_mgr.slope_length == 0) and self.average_plateaus:
            x = self.waveform_mgr.plateaus_to_points(x)
        return x

    def format_targets(self, x: torch.Tensor) -> torch.Tensor:
        """ 
        The hardware processor uses a waveform to represent 
        points (see 5.1 in Introduction of the Wiki). Each point is represented with some
        slope and some plateau points. When passing through the hardware, there will be a
        difference between the output from the device and the input (in points). This function
        is used for the targets to have the same length in shape as the outputs. It simply 
        repeats each point in the input as many times as there are points in the plateau. In 
        this way, targets can then be compared against hardware outputs in the loss function.

        Parameters
        ----------
        x : torch.Tensor
            Targets of the supervised learning problem, that will be extended to have the same
            length shape as the outputs from the processor.
        """
        if not (self.waveform_mgr.plateau_length == 1
                and self.waveform_mgr.slope_length
                == 0) and not self.average_plateaus:
            x = self.waveform_mgr.points_to_plateaus(x)

        return x

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


def get_electrode_info(configs):
    """
    Retrieve electrode information from the data sampling configurations.

    Parameters
    ----------
    configs: dict
        driver:
            amplification: Amplification correction value of the output. Calculated from the op-amp.
            instruments_setup:
                multiple_devices:
                activation_channels
                activation_voltage_ranges
                readout_channels

    Returns
    -------
    electrode_info : dict
        Configuration dictionary containing all the keys related to the electrode information:
            * electrode_no: int
                Total number of electrodes in the device
            * activation_electrodes: dict
                - electrode_no: int
                    Number of activation electrodes used for gathering the data
                - voltage_ranges: list
                    Voltage ranges used for gathering the data. It contains the ranges per
                    electrode, where the shape is (electrode_no,2). Being 2 the minimum and maximum
                    of the ranges, respectively.
            * output_electrodes: dict
                - electrode_no : int
                    Number of output electrodes used for gathering the data
                - clipping_value: list[float,float]
                    Value used to apply a clipping to the sampling data within the specified values.
                - amplification: float
                    Amplification correction factor used in the device to correct the amplification
                    applied to the output current in order to convert it into voltage before its
                    readout.
    """
    electrode_info = {}
    assert not configs['driver']['instruments_setup'][
        'multiple_devices'], "A single processor does not support multiple DNPUs."
    activation_electrode_no = len(
        configs['driver']['instruments_setup']['activation_channels'])
    readout_electrode_no = len(
        configs['driver']['instruments_setup']['readout_channels'])

    electrode_info["electrode_no"] = (activation_electrode_no +
                                      readout_electrode_no)
    electrode_info["activation_electrodes"] = {}
    electrode_info["activation_electrodes"][
        "electrode_no"] = activation_electrode_no
    electrode_info["activation_electrodes"]["voltage_ranges"] = configs[
        'driver']['instruments_setup']['activation_voltage_ranges']
    electrode_info["output_electrodes"] = {}
    electrode_info["output_electrodes"]["electrode_no"] = readout_electrode_no
    electrode_info["output_electrodes"]["amplification"] = configs["driver"][
        "amplification"]
    electrode_info["output_electrodes"]["clipping_value"] = [
        -float("Inf"), float("Inf")
    ]

    return electrode_info
