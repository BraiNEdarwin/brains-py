import numpy as np
import warnings
from brainspy.processors.hardware.drivers.ni.setup import (
    NationalInstrumentsSetup,
    SYNCHRONISATION_VALUE,
    CDAQ_TO_NIDAQ_RAMPING_TIME_SECONDS,
)


class CDAQtoNiDAQ(NationalInstrumentsSetup):
    """
    Class to establish a connection (for a single, or multiple hardware DNPUs) with the CDAQtoNiDAQ
    national instrument. It requires an additional channel to send a spike
    from the CDAQ to the NIDAQ. The data is offsetted to let the NIDAQ read the spike and start
    synchronising after receiving it.
    """
    def __init__(self, configs):
        """
        Initialize the hardware processor. No trigger source required for this device.

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
                Value (in milliseconds) that the original
                activation voltage will be displaced, in order to enable the spiking signal to
                reach the nidaq setup. The default value is the SYNCHRONISATION_VALUE multiplied
                by the activation instrument sampling frequency.

            max_ramping_time_seconds : int
                To set the ramp time for the setup. It is defined with the flags
                CDAQ_TO_CDAQ_RAMPING_TIME_SECONDS in
                brainspy/processors/hardware/drivers/ni/setup.py. Do not tamper with it,
                as it could disable security checks designed to avoid breaking devices.

        """
        configs["auto_start"] = False
        configs["offset"] = int(
            configs["instruments_setup"]["activation_sampling_frequency"] *
            SYNCHRONISATION_VALUE)
        configs[
            "max_ramping_time_seconds"] = CDAQ_TO_NIDAQ_RAMPING_TIME_SECONDS
        super().__init__(configs)
        self.tasks_driver.add_synchronisation_channels(
            self.configs["instruments_setup"]["readout_instrument"],
            self.configs["instruments_setup"]["activation_instrument"],
        )

    def forward_numpy(self, y):
        """
        The forward function computes output numpy values from input numpy array.
        This is done to enable compatibility of the the model with numpy
        The first point of the read_data does not perform a reading.
        To synchronise it with the original signal, a point is added at the original signal y.
        The signal read in 'data' discards the first point.

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
        self.original_shape = y.shape[0]
        y = y.T

        # assert (self.configs["data"]["shape"] == y.shape[1]
        #         ), f"configs value with key 'shape' must be {y.shape[1]}"
        y = self.synchronise_input_data(y)
        max_attempts = 5
        attempts = 1
        finished = False
        while not finished and (attempts < max_attempts):
            data, finished = self.readout_trial(y)
            attempts += 1
        data *= self.inversion
        assert finished, (
            "Error: unable to synchronise input and output. Output: " +
            str(data.shape[1]) + " points.")
        return data.T

    def readout_trial(self, y):
        """
        Attempts to perform a readout from the device given an input array.
        Reads the data, processes it and synchronises the output with regard
        to the input data.

        Parameters
        ----------
        y : np.array
            Input data matrix to be sent to the device.
            The data should have a shape of: (device_input_channel_no, data_point_no)
            Where device_input_channel_no is typically the number of activation
            electrodes of the DNPU.

        Returns
        -------
        np.array,bool
            Synchronised output data from the device and wheather the readout is complete
        """
        data = self.read_data(y)
        data = self.process_output_data(data)
        data = self.synchronise_output_data(data)
        finished = data.shape[1] == self.original_shape
        return data, finished

    def synchronise_input_data(self, y):
        """
        The input signal is synchronised with the output sending a spike through a synchronisation
        channel. In order to wait for the output reading module to receive the spike, zeros are
        added to the input data up until the point where the spike should have been received. This
        is done by adding an offset.

        Parameters
        ----------
        y : np.array
            Input data to be sent to the device.

        Returns
        -------
        np.array
            Synchronised input data based on the offset value, where the synchronisation spike
            should have been received.
        """
        # TODO: Are the following three lines really necessary?
        y = np.asarray(y)
        if len(y.shape) == 1:
            y = y[np.newaxis, :]
        # Append some zeros to the initial signal such that no input data is lost
        # This should be handled with proper synchronization
        y_corr = np.zeros((
            y.shape[0], y.shape[1] +
            self.configs["offset"]))  # Add 200ms of reaction in terms of zeros
        y_corr[:, self.configs["offset"]:] = y[:]
        # TODO: Is this if really necessary?
        if len(y_corr.shape) == 1:
            y_corr = np.concatenate(
                (y_corr[np.newaxis], np.zeros(
                    (1, y_corr.shape[1]))))  # Set the trigger
        else:
            y_corr = np.concatenate((y_corr, np.zeros(
                (1, y_corr.shape[1]))))  # Set the trigger
        y_corr[-1, self.configs["offset"]] = 1  # Start input data

        return y_corr

    def get_output_cut_value(self, read_data):
        """
        The input signal is synchronised with the output sending a spike through a synchronisation
        channel. This method gets the value where the output data should be cut in order to make
        the output signal be synchronised with regard to the input signal.

        Parameters
        ----------
        read_data : np.array
            Processed output data computed from the amplification value.

        Returns
        -------
        int
            Output cut value
        """
        cut_value = np.argmax(read_data[-1, :])
        if read_data[-1, cut_value] < 0.05:
            warnings.warn("initialize spike not recognised")
        return cut_value

    def synchronise_output_data(self, read_data):
        """
        The input signal is synchronised with the output sending a spike through a synchronisation
        channel. All the data before the output reading instrument receives the spike is discarded.
        This method cuts the output data in order to make the input signal be synchronised with the
        output signal.

        Parameters
        ----------
        read_data : np.array
            processed output data computed from the amplification value

        Returns
        -------
        np.array
            synchronized output data
        """
        cut_value = self.get_output_cut_value(read_data)
        return read_data[:-1, cut_value:self.original_shape + cut_value]
