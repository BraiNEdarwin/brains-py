import numpy as np

from brainspy.processors.hardware.drivers.ni.setup import (
    NationalInstrumentsSetup,
    CDAQ_TO_CDAQ_RAMPING_TIME_SECONDS,
)


class CDAQtoCDAQ(NationalInstrumentsSetup):
    """
    Class to establish a connection (for a single, or multiple hardware DNPUs) with the
    CDAQ-to-CDAQ national instrument. It can be of 2 types:
            * With a regular rack
            * With a real time rack
    """
    def __init__(self, configs):
        """
        Initialize the hardware processor

        Parameters
        ----------
        configs : dict
            Key-value pairs required in the configs dictionary to initialise the driver. These are
            described in the parent class
            brainspy.processors.hardware.drivers.ni.setup.NationalInstrumentsSetup.
            Appart from the values described there, there are some internal keys that are added
            internally in this class during the initialisation. None of these are required to
            be passed on the configs.

            auto_start : bool
                If the task is not explicitly started with the DAQmx start_task method, it will
                start it anyway. This value is set to True for this setup.

            offset : int
                Only for CDAQ TO NIDAQ setup. Value (in milliseconds) that the original
                activation voltage will be displaced, in order to enable the spiking signal to
                reach the nidaq setup. The offset value is set to 1 for this setup.

            max_ramping_time_seconds : int
                To set the ramp time for the setup. It is defined with the flags
                CDAQ_TO_CDAQ_RAMPING_TIME_SECONDS in
                brainspy/processors/hardware/drivers/ni/setup.py. Do not tamper with it,
                as it could disable security checks designed to avoid breaking devices.
        """
        configs["auto_start"] = True
        configs["offset"] = int(10000/configs['DAC_update_rate'])
        configs["max_ramping_time_seconds"] = CDAQ_TO_CDAQ_RAMPING_TIME_SECONDS
        super().__init__(configs)
        self.tasks_driver.start_trigger(
            self.configs["instruments_setup"]["trigger_source"])

    def forward_numpy(self, y):
        """
        The forward function computes output numpy values from input numpy array.
        This is done to enable compatibility of the the model with numpy
        The first point of the read_data does not perform a reading.
        To synchronise it with the original signal, a point is added at the original signal y.
        The signal read in 'data' discards the first point

        Parameters
        ----------
        y : np.array
            Input data matrix to be sent to the device.
            The data should have a shape of: (device_input_channel_no, data_point_no)
            Where device_input_channel_no is typically the number of activation
            electrodes of the DNPU.

        Returns
        -------
        np.array
            Output data that has been read from the device when receiving the input y.
        """

        #y = np.concatenate((y, y[-1, :] * np.ones((1, y.shape[1]))))
        y = y.T
        data = self.read_data(y)
        data = -1 * self.process_output_data(data)[:,int((10000/self.configs['DAC_update_rate'])):] #-1 * self.process_output_data(data)#[:, 1:]
        return data.T
