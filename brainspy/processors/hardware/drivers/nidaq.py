import numpy as np
import warnings
from brainspy.processors.hardware.drivers.ni.setup import (
    NationalInstrumentsSetup,
    SYNCHRONISATION_VALUE,
    CDAQ_TO_NIDAQ_RAMPING_TIME_SECONDS,
)


class CDAQtoNiDAQ(NationalInstrumentsSetup):
    """
    Class to establish a connection (for a single, or multiple hardware DNPUs) with the CDAQtoNiDAQ national instrument
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
                        activation_instrument: str - name of the activation instrument
                        activation_channels: list - [8,10,13,11,7,12,14] - Channels through which voltages will be sent for activating the device (with both data inputs and control voltages)
                        min_activation_voltages: list - list - [-1.2, -1.2, -1.2, -1.2, -1.2, -0.7, -0.7] - minimum value for each of the activation electrodes
                        max_activation_voltages: list -[0.6, 0.6, 0.6, 0.6, 0.6, 0.3, 0.3] - maximum value for each of the activation electrodes
                        readout_instrument: str -name of the readout instrument
                        readout_channels: [2] list - Channels for reading the output current values
                        trigger_source: str - source trigger name
        """
        configs["auto_start"] = False
        configs["offset"] = int(
            configs["driver"]["sampling_frequency"] * SYNCHRONISATION_VALUE
        )
        configs["max_ramping_time_seconds"] = CDAQ_TO_NIDAQ_RAMPING_TIME_SECONDS
        super().__init__(configs)
        self.tasks_driver.add_channels(
            self.configs["driver"]["readout_instrument"],
            self.configs["driver"]["activation_instrument"],
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
        y = y.T
        assert (
            self.configs["data"]["shape"] == y.shape[1]
        ), f"configs value with key 'shape' must be {y.shape[1]}"
        y = self.synchronise_input_data(y)
        max_attempts = 5
        attempts = 1
        finished = False
        while not finished and (attempts < max_attempts):
            data, finished = self.readout_trial(y)
            attempts += 1

        assert finished, (
            "Error: unable to synchronise input and output. Output: "
            + str(data.shape[1])
            + " points."
        )
        return data.T

    def readout_trial(self, y):
        """
        Readout data from the device.
        Reads the data, processes it and synchronises the output data.

        Parameters
        ----------
        y : np.array
            It represents the output data as matrix

        Returns
        -------
        np.array,bool
            synchronised output data from the device and wheather the readout is complete
        """
        data = self.read_data(y)
        data = self.process_output_data(data)
        data = self.synchronise_output_data(data)
        finished = data.shape[1] == self.configs["data"]["shape"]
        return data, finished

    def synchronise_input_data(self, y):
        """
        Synchronize the input data to feed the device based on the offset value

        Parameters
        ----------
        y : np.array
            It represents the input data as matrix where the shpe is defined by
            the "number of inputs to the device" times "input points that you want to input to the device".

        Returns
        -------
        np.array
            synchronized input data based on the offset value
        """
        # TODO: Are the following three lines really necessary?
        y = np.asarray(y)
        if len(y.shape) == 1:
            y = y[np.newaxis, :]
        # Append some zeros to the initial signal such that no input data is lost
        # This should be handled with proper synchronization
        y_corr = np.zeros(
            (y.shape[0], y.shape[1] + self.configs["offset"])
        )  # Add 200ms of reaction in terms of zeros
        y_corr[:, self.configs["offset"] :] = y[:]
        # TODO: Is this if really necessary?
        if len(y_corr.shape) == 1:
            y_corr = np.concatenate(
                (y_corr[np.newaxis], np.zeros((1, y_corr.shape[1])))
            )  # Set the trigger
        else:
            y_corr = np.concatenate(
                (y_corr, np.zeros((1, y_corr.shape[1])))
            )  # Set the trigger
        y_corr[-1, self.configs["offset"]] = 1  # Start input data

        return y_corr

    def get_output_cut_value(self, read_data):
        """
        get the output cut value from the processed output data
        cut-off values are the dividing points on the output data that divides them into different categories

        Parameters
        ----------
        read_data : np.array
            processed output data computed from the amplification value

        Returns
        -------
        int
            output cut value
        """
        cut_value = np.argmax(read_data[-1, :])
        if read_data[-1, cut_value] < 0.05:
            warnings.warn("initialize spike not recognised")
        return cut_value

    def synchronise_output_data(self, read_data):
        """
        Synchronize th output data from the device

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
        return read_data[:-1, cut_value : self.configs["data"]["shape"] + cut_value]