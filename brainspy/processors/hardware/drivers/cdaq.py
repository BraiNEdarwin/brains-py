import numpy as np

from brainspy.processors.hardware.drivers.ni.setup import (
    NationalInstrumentsSetup,
    CDAQ_TO_CDAQ_RAMPING_TIME_SECONDS,
)


class CDAQtoCDAQ(NationalInstrumentsSetup):
    """
    Class to establish a connection (for a single, or multiple hardware DNPUs) with the CDAQ-to-CDAQ national instrument
    It can be of 2 types :
            * With a regular rack
            * With a real time rack
    """

    def __init__(self, configs):
        """
        Initialize the hardware processor

        Parameters
        ----------
        configs : dict
        Data key,value pairs required in the configs to initialise the hardware processor :

            max_ramping_time_seconds : int - To set the ramp time for the setup
                                            WARNING -The security check for the ramping time has been disabled. Steep rampings can can damage the device.
            offset : int - To set the offset value of the wave
            auto_start : bool - Too auto start the setup tasks or not
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
        """
        configs["auto_start"] = True
        configs["offset"] = 1
        configs["max_ramping_time_seconds"] = CDAQ_TO_CDAQ_RAMPING_TIME_SECONDS
        super().__init__(configs)
        self.tasks_driver.start_trigger(
            self.configs["instruments_setup"]["trigger_source"]
        )

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
            input data

        Returns
        -------
        np.array
            output data
        """

        y = np.concatenate((y, y[-1, :] * np.ones((1, y.shape[1]))))
        y = y.T
        data = self.read_data(y)
        data = -1 * self.process_output_data(data)[:, 1:]
        return data.T
