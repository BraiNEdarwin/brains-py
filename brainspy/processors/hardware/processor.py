import torch
import warnings
from torch import nn
from brainspy.utils.manager import get_driver
from brainspy.utils.pytorch import TorchUtils
from brainspy.utils.waveform import WaveformManager


class HardwareProcessor(nn.Module):
    """
    The HardwareProcessor class establishes a connection (for a single, or multiple hardware DNPUs) with one of the following National Instruments measurement devices:
                * CDAQ-to-NiDAQ
                * CDAQ-to-CDAQ
                        * With a regular rack
                        * With a real time rack

    """

    # TODO: Automatically register the data type according to the configurations of the amplification variable of the  info dictionary
    # Instrument configs can be a dictionary or a driver which has been already initialised
    def __init__(self, instrument_configs, slope_length, plateau_length, logger=None):
        """
        To intialise the hardware processor

        Parameters
        ----------
        configs : dict
        Data key,value pairs required in the configs to initialise the hardware processor :

            processor_type : str - "simulation_debug" or "cdaq_to_cdaq" or "cdaq_to_nidaq" - Processor type to initialize a hardware processor
            max_ramping_time_seconds : int - To set the ramp time for the setup
            data:
                waveform:
                    plateau_length: float - A plateau of at least 3 is needed to train the perceptron (That requires at least 10 values (3x4 = 12)).
                    slope_length : float - Length of the slope of a waveform
                activation_electrode_no: int - It specifies the number of activation electrodes. Only required for simulation mode
            driver:
                instruments_setup:
                    device_no: str - "single" or "multiple" - depending on the number of devices
                    activation_instrument: str - cDAQ1Mod3 - name of the activation instrument
                    activation_channels: list - [8,10,13,11,7,12,14] - Channels through which voltages will be sent for activating the device (with both data inputs and control voltages)
                    min_activation_voltages: list - list - [-1.2, -1.2, -1.2, -1.2, -1.2, -0.7, -0.7] - minimum value for each of the activation electrodes
                    max_activation_voltages: list -[0.6, 0.6, 0.6, 0.6, 0.6, 0.3, 0.3] - maximum value for each of the activation electrodes
                    readout_instrument: str cDAQ1Mod4 - name of the readout instrument
                    readout_channels: [2] list - Channels for reading the output current values
                    trigger_source: str - cDAQ1/segment1 - source trigger name
                tasks_driver_type : str - "local" or "remote" - type of tasks driver
                amplification: float - The output current (nA) of the device is converted by the readout hardware to voltage (V), because it is easier to do the readout of the device in voltages.
                This output signal in nA is amplified by the hardware when doing this current to voltage conversion, as larger signals are easier to detect.
                In order to obtain the real current (nA) output of the device, the conversion is automatically corrected in software by multiplying by the amplification value again.
                The amplification value depends on the feedback resistance of each of the setups. Below, there is a guide of the amplification value needed for each of the setups:

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

                                        If no correction is desired, the amplification can be set to 1.

                sampling_frequency: int - the average number of samples to be obtained in one second
                output_clipping_range: [float,float] - The the setups have a limit in the range they can read. They typically clip at approximately +-4 V.
                    Note that in order to calculate the clipping_range, it needs to be multiplied by the amplification value of the setup. (e.g., in the Brains setup the amplification is 28.5,
                    is the clipping_value is +-4 (V), therefore, the clipping value should be +-4 * 28.5, which is [-110,110] (nA) ).
                    The original clipping value of the surrogate models is obtained when running the preprocessing of the data in
                    bspysmg.measurement.processing.postprocessing.post_process.

        logger : logging , optional
            To emit log messages at different levels(DEBUG, INFO, ERROR, etc.)
            It provides a way for applications to configure different log handlers , by default None
        """
        super(HardwareProcessor, self).__init__()

        if not isinstance(instrument_configs, dict):
            self.driver = instrument_configs
        else:
            self.driver = get_driver(instrument_configs)
            self.voltage_ranges = TorchUtils.format(self.driver.voltage_ranges)
            # TODO: Add message for assertion. Raise an error.
            assert (
                slope_length / self.driver.configs["sampling_frequency"]
            ) >= self.driver.configs["max_ramping_time_seconds"]

        self.waveform_mgr = WaveformManager(
            {"slope_length": slope_length, "plateau_length": plateau_length}
        )

        self.logger = logger

    def forward(self, x):
        """
        The forward function computes output Tensors from input Tensors.
        This is done to enable compatibility of the the model ,which is an nn.Module, with Pytorch
        There is an additon of slopes to the data in this model. The data in pytorch is represented in the form of plateaus.

        Parameters
        ----------
        x : torch.Tensor
            input data

        Returns
        -------
        torch.Tensor
            output data

        """
        with torch.no_grad():
            x, mask = self.waveform_mgr.plateaus_to_waveform(x, return_pytorch=False)
            output = self.forward_numpy(x)
            if self.logger is not None:
                self.logger.log_output(x)
        return TorchUtils.format(output[mask])

    def forward_numpy(self, x):
        """
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
        return self.driver.forward_numpy(x)

    def reset(self):
        """
        To reset the driver
        """
        self.driver.reset()

    def close(self):
        """
        To close the driver if specified in the driver directory or raise a warning if the driver has not been closed after use.
        """
        if "close_tasks" in dir(self.driver):
            self.driver.close_tasks()
        else:
            warnings.warn("Driver tasks have not been closed.")

    def is_hardware(self):
        """
        To check if the driver is a hardware or not

        Returns
        -------
        bool
            True or False depending on wheather it is a hardware or software driver
        """
        return self.driver.is_hardware()

    # def get_electrode_no(self):
    #     """
    #     To get the electrode number that is being used by this driver

    #     Returns
    #     -------
    #     int
    #         the electrode number
    #     """
    #     return self.electrode_no