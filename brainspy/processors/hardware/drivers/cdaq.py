
import numpy as np

from brainspy.processors.hardware.drivers.ni.setup import NationalInstrumentsSetup, CDAQ_TO_CDAQ_RAMPING_TIME_SECONDS


class CDAQtoCDAQ(NationalInstrumentsSetup):
    def __init__(self, configs):
        configs["auto_start"] = True
        configs["offset"] = 1
        configs["max_ramping_time_seconds"] = CDAQ_TO_CDAQ_RAMPING_TIME_SECONDS
        super().__init__(configs)
        self.driver.start_trigger(self.configs["driver"]["trigger_source"])

    def forward_numpy(self, y):
        # The first point of the read_data does not perform a reading.
        # To synchronise it with the original signal, a point is added at the original signal y.
        # The signal read in 'data' discards the first point

        y = np.concatenate((y, y[-1, :] * np.ones((1, y.shape[1]))))
        y = y.T
        data = self.read_data(y)
        data = -1 * self.process_output_data(data)[:, 1:]
        return data.T
