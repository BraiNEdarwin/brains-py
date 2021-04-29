import numpy as np

from brainspy.processors.hardware.drivers.ni.setup import (
    NationalInstrumentsSetup,
    SYNCHRONISATION_VALUE,
    CDAQ_TO_NIDAQ_RAMPING_TIME_SECONDS,
)


class CDAQtoNiDAQ(NationalInstrumentsSetup):
    def __init__(self, configs):
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
            "Error: unable to synchronise input and output. Output: " + str(data.shape[1]) + " points."
        )
        return data.T

    def readout_trial(self, y):
        data = self.read_data(y)
        data = self.process_output_data(data)
        data = self.synchronise_output_data(data)
        finished = data.shape[1] == self.configs["data"]["shape"]
        return data, finished

    def synchronise_input_data(self, y):
        # TODO: Are the following three lines really necessary?
        y = np.asarray(y)
        if len(y.shape) == 1:
            y = y[np.newaxis, :]
        # Append some zeros to the initial signal such that no input data is lost
        # This should be handled with proper synchronization
        y_corr = np.zeros(
            (y.shape[0], y.shape[1] + self.configs["offset"])
        )  # Add 200ms of reaction in terms of zeros
        y_corr[:, self.configs["offset"]:] = y[:]
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
        cut_value = np.argmax(read_data[-1, :])
        if read_data[-1, cut_value] < 0.05:
            print("Warning: initialize spike not recognised")
        return cut_value

    def synchronise_output_data(self, read_data):
        cut_value = self.get_output_cut_value(read_data)
        return read_data[:-1, cut_value: self.configs["data"]["shape"] + cut_value]
