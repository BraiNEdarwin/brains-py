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
                Number of points that the original activation voltage signal will be displaced, in
                order to enable the spiking signal to reach the nidaq setup. The offset value is
                set to 0 for this setup.

            max_ramping_time_seconds : int
                To set the ramp time for the setup. It is defined with the flags
                CDAQ_TO_CDAQ_RAMPING_TIME_SECONDS in
                brainspy/processors/hardware/drivers/ni/setup.py. Do not tamper with it,
                as it could disable security checks designed to avoid breaking devices.
        """
        assert type(
            configs) == dict, "The configurations should be of type - dict"
        configs["auto_start"] = True
        configs["offset"] = 1
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
        assert type(
            y) == np.ndarray, "The input should be of type -numpy array"
        # The convention for pytorch and nidaqmx is different. Therefore,
        # the input to the device needs to be transposed before sending it to the device.
        y = y.T
        data = self.read_data(y)

        # Convert list to numpy, ensure it has dimension (channel_no, data)
        data = self.process_output_data(data)

        # The CDAQ measurements always add an extra point at the beginning of the
        # measurement. The following line removes it before applying averaging
        # It also applies an inversion (when applicable) to the averaged output.

        data = self.inversion * self.average_point_difference(
            data[:, self.configs['offset']:])

        # The convention for pytorch and nidaqmx is different. Therefore,
        # the output from the device needs to be transposed before sending it back.
        return data.T
