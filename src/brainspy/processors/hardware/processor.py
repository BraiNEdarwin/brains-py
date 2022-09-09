import torch
import warnings
from torch import nn
import numpy as np
from brainspy.utils.manager import get_driver
from brainspy.utils.pytorch import TorchUtils
from brainspy.utils.waveform import WaveformManager


class HardwareProcessor(nn.Module):
    """
    The HardwareProcessor class helps handling the data before sending it to the drivers of the
    hardware setups.The input data to the hardware drivers has to be given with a waveform.
    The waveform is composed of slopes and plateaus.The points in the data that is passed to the
    HardwareProcessor need to be already represented as plateaus. The HardwareProcessor
    class creates the slopes to the data, in form of pytorch Tensors, and transforms it to numpy
    arrays before sending it to the driver.It transforms the obtained readout result back to pytorch
    Tensors, and removes the created rampings. The input and output Tensors have the same length.
    The drivers can establish a connection (for a single, or multiple hardware DNPUs) with one of
    the following National Instruments measurement devices:
                * CDAQ-to-NiDAQ
                * CDAQ-to-CDAQ
                        * With a regular rack
                        * With a real time rack
    Please check https://github.com/BraiNEdarwin/brains-py/wiki/A.-Introduction
    for more information about hardware setups and how the waveform works.
    """

    # TODO: Automatically register the data type according to the configurations of the
    #  amplification variable of the info dictionary
    # Instrument configs can be a dictionary or a driver which has been already initialised
    def __init__(self,
                 instrument_configs,
                 slope_length,
                 plateau_length):
        """
        To intialise the hardware processor

        Parameters
        ----------
        instruments_configs : dict or SurrogateModel

            - If a SurrogateModel instance is provided, it will simulate a hardware processor
            for debugging purposes, without connecting to real hardware.
            Refer to brainspy.simulation.processor to see how a SurrogateModel can be defined.

            - If the instruments configs are provided as a dict, the configs should have the following keys:

                inverted_output : bool
                    True if inversion should be applied to the output of the DNPU, else False.
                amplification: float
                    The output current (nA) of the device is converted by the readout hardware to
                    voltage (V), because it is easier to do the readout of the device in voltages.
                    This output signal in nA is amplified by the hardware when doing this current
                    to voltage conversion, as larger signals are easier to detect. In order to
                    obtain the real current (nA) output of the device, the conversion is
                    automatically corrected in software by multiplying by the amplification value
                    again.
                    The amplification value depends on the feedback resistance of each of the
                    setups.

                        Below, there is a guide of the amplification value needed for each of the
                        setups:

                                            Darwin: Variable amplification levels:
                                                A: 1000 Amplification
                                                Feedback resistance: 1 MOhm
                                                B: 100 Amplification
                                                Feedback resistance 10 MOhms
                                                C: 10 Amplification
                                                Feedback resistance: 100 MOhms
                                                D: 1 Amplification
                                                Feedback resistance 1 GOhm
                                            Pinky:  - PCB 1 (6 converters with):
                                                    Amplification 10
                                                    Feedback resistance 100 MOhm
                                                    - PCB 2 (6 converters with):
                                                    Amplification 100 tims
                                                    10 mOhm Feedback resistance
                                            Brains: Amplfication 28.5
                                                    Feedback resistance, 33.3 MOhm
                                            Switch: (Information to be completed)
                                            If no correction is desired, the amplification can be set
                                            to 1.
                    instruments_setup:
                        multiple_devices: boolean
                            False will initialise the drivers to read from a single hardware DNPU.
                            True, will enable to read from more than one DNPU device at the same time.
                        activation_instrument: str
                            Name of the activation instrument as observed in the NI Max software.
                            E.g.,  cDAQ1Mod3
                        activation_sampling_frequency: int
                            The number of samples to be obtained in one second,
                            when transforming the activation signal from digital to analogue.
                        activation_channels: list
                            Channels through which voltages will be sent for activating the device
                            (both data inputs and control voltage electrodes). The channels can be
                            checked in the schematic of the DNPU device.
                            E.g., [8,10,13,11,7,12,14]
                        activation_voltage_ranges: list
                            Minimum and maximum voltage for the activation electrodes.
                            E.g., [[-1.2, 0.6], [-1.2, 0.6],
                            [-1.2, 0.6], [-1.2, 0.6], [-1.2, 0.6], [-0.7, 0.3], [-0.7, 0.3]]
                        readout_instrument: str
                            Name of the readout instrument as observed in the NI Max software.
                            E.g., cDAQ1Mod4
                        readout_sampling_frequency: int
                            The number of samples to be obtained in one second, when transforming
                            the readout signal from analogue to digital.
                        readout_channels: [2] list
                            Channels for reading the output current values.
                            The channels can be checked in the schematic of the DNPU device.
                        trigger_source: str
                            For synchronisation purposes, sending data for the activation voltages on
                            one NI Task can trigger the readout device of another NI Task. In these
                            cases,the trigger source name should be specified in the configs.
                            This is only applicable for CDAQ to CDAQ setups
                            (with or without real-time rack).
                            E.g., cDAQ1/segment1
                            More information at
                            https://nidaqmx-python.readthedocs.io/en/latest/start_trigger.html

            plateau_length: float - Length of the plateau that is being sent through the forward
            call of the HardwareProcessor
            slope_length : float - Length of the slopes in the waveforms sent to the device through
            the drivers

            The input data to the hardware drivers has to be given with a waveform. The waveform is
            composed of slopes and plateaus.
            Please check https://github.com/BraiNEdarwin/brains-py/wiki/A.-Introduction for more
            information about hardware setups and how the waveform works.

        """
        super(HardwareProcessor, self).__init__()
        assert type(plateau_length) == int or type(
            plateau_length
        ) == float, "The plateau length should be of type -int or float"
        assert type(slope_length) == int or type(
            slope_length
        ) == float, "The slope length should be of type - int or float"
        if not isinstance(instrument_configs, dict):
            self.driver = instrument_configs
            #if self.driver.is_hardware():
            self.voltage_ranges = self.driver.get_voltage_ranges()
            # else:
            #     self.voltage_ranges = None
            self.clipping_value = self.driver.get_clipping_value()
        else:
            self.driver = get_driver(instrument_configs)
            self.register_buffer(
                "voltage_ranges",
                torch.tensor(self.driver.voltage_ranges,
                             dtype=torch.get_default_dtype()))
            self.clipping_value = None
            assert ( (slope_length / self.driver.configs["instruments_setup"][
                    "activation_sampling_frequency"]) <= self.driver.configs[
                        "max_ramping_time_seconds"]), "The ratio of the slope length and the activation sampling frequency cannot be less than the max ramping time"

        self.waveform_mgr = WaveformManager({
            "slope_length": slope_length,
            "plateau_length": plateau_length
        })

    def forward(self, x):
        """
        The forward function sends the data through the driver and returns the information from
        hardware DNPUs. It receives an input pytorch Tensor, where points are represented in
        plateaus. Then, it generates according rampings to the plateaus to obtain a waveform,
        translates the result into a numpy array and sends the data to the driver. The resulting
        response from the DNPU hardware is then converted back to a pytorch tensor, and the
        rampings, where the results of the rampings are filtered out. The output pytorch Tensor
        will have the same length as the input pytorch Tensor.
        The purpose of this class is to compatibility of the the model with Pytorch, and make
        possible to seamlessly exchange simulations of DNPUs with real hardware.
        There is an additon of slopes to the data in this model. The data in pytorch is represented
        in the form of plateaus.

        Parameters
        ----------
        x : torch.Tensor
            input data in 'plateau' format (the forward pass will add/remove the slopes to the data).
            The expected shape is (batch_size, activation_electrode_no)

        Returns
        -------
        torch.Tensor
            output data

        """
        assert type(
            x) == torch.Tensor, "The input should be of type - torch.Tensor"
        assert x.shape[-1] == len(
            self.driver.configs['instruments_setup']['activation_channels'])
        with torch.no_grad():
            device, dtype = x.device, x.dtype
            x, mask = self.waveform_mgr.plateaus_to_waveform(
                x, return_pytorch=False)
            if len(x.shape) > 2:
                x = x.squeeze()
            x = self.forward_numpy(x)
        return TorchUtils.format(x[mask], device=device, data_type=dtype)

    def forward_numpy(self, x):
        """
        It enables to use directly the driver, without any transformation to the data. The input
        should already be in the form of a waveform.
        The output will be returned in the form of a waveform with the same length as the input.
        The forward function computes output numpy values from input numpy array.
        This is done to enable compatibility of the the model which is an nn.Module with numpy

        Parameters
        ----------
        x : np.array
            input data

        Returns
        -------
        np.array
            output data

        """
        assert type(
            x) == np.ndarray, "The input data should be of type - numpy array"
        return self.driver.forward_numpy(x)

    def close(self):
        """
        Closes the driver if specified in the driver directory or raise a warning if the driver has
        not been closed after use.
        """
        if "close_tasks" in dir(self.driver):
            self.driver.close_tasks()
        else:
            warnings.warn(
                "It was not possible to close the NI Tasks from the driver." +
                "This should be fine if you are running a simulation.")

    def is_hardware(self):
        """
        Checks if the driver is a hardware or not. It will return True if the driver is a
        NationalInstrumentsSetup instance. It will return False if the HardwareProcessor is
        initialised with the 'simulation_debug' flag in the 'processor_type' configuration.

        Returns
        -------
        bool
            True or False depending on wheather it is a hardware or software driver
        """
        return self.driver.is_hardware()

    def get_voltage_ranges(self):
        """
        Gets the voltage ranges declared on the hardware processor.

        Returns
        -------
        voltage_ranges
            torch.tensor
        """
        return self.voltage_ranges

    def get_clipping_value(self):
        """
        Gets the clipping value declared on the hardware processor.

        Only exists if processor is for simulation debug.
        Will be None if processor is hardware.

        Returns
        -------
        torch.Tensor or None
            Clipping value.
        """
        return self.clipping_value
