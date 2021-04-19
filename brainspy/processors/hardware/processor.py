""" 
The accelerator class enables to statically access the accelerator
(CUDA or CPU) that is used in the computer. The aim is to support both platforms seemlessly.
"""

import torch
import warnings
from torch import nn
from brainspy.utils.manager import get_driver
from brainspy.utils.pytorch import TorchUtils
from brainspy.utils.waveform import WaveformManager
from brainspy.processors.simulation.processor import SurrogateModel


class HardwareProcessor(nn.Module):
    """
    The HardwareProcessor class establishes a connection (for a single, or multiple hardware DNPUs) with one of the following National Instruments measurement devices:
                * CDAQ-to-NiDAQ
                * CDAQ-to-CDAQ
                        * With a regular rack
                        * With a real time rack

    The TorchModel class is used to manage together a torch model and its state dictionary.
    Usage example :
        mymodel = TorchModel()
        mymodel.load_model('my_path/my_model.pt')
        mymodel.model
    """

    # TODO: Automatically register the data type according to the configurations of the amplification variable of the  info dictionary
    def __init__(self, configs, logger=None):
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
                amplification: float - To set the amplification value of the voltages
                sampling_frequency: int - the average number of samples to be obtained in one second
                output_clipping_range: [float,float] - To clip the output voltage if it goes above maximum

        logger : logging , optional
            To emit log messages at different levels(DEBUG, INFO, ERROR, etc.)
            It provides a way for applications to configure different log handlers , by default None
        """
        super(HardwareProcessor, self).__init__()
        self.driver = get_driver(configs)
        if configs["processor_type"] == "simulation_debug":
            self.voltage_ranges = self.driver.voltage_ranges
        else:
            self.voltage_ranges = TorchUtils.format(self.driver.voltage_ranges)
        self.waveform_mgr = WaveformManager(configs["data"]["waveform"])
        self.logger = logger
        self.amplification = configs["driver"]["amplification"]
        self.clipping_value = [
            configs["driver"]["output_clipping_range"][0] * self.amplification,
            configs["driver"]["output_clipping_range"][1] * self.amplification,
        ]
        self.electrode_no = configs["data"]["activation_electrode_no"]

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
            raise Warning("Driver tasks have not been closed.")

    def is_hardware(self):
        """
        To check if the driver is a hardware or not

        Returns
        -------
        bool
            True or False depending on wheather it is a hardware or software driver
        """
        return self.driver.is_hardware()

    def get_electrode_no(self):
        """
        To get the electrode number that is being used by this driver

        Returns
        -------
        int
            the electrode number
        """
        return self.electrode_no
