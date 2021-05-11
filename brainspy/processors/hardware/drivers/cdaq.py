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

            max_ramping_time_seconds : int - The ramping time for the setup of the this device.The Ramp Time is used to designate the amount of time it will take to ramp to a pressure.
                                            WARNING -The security check for the ramping time has been disabled. Steep rampings can can damage the device.

            offset : int - To set the offset value of the wave.
                            This done by adding a number to a signal performs an offset. The addition shifts the value of every sample up (or down) by the same amount.

            auto_start : bool - Too auto start the setup tasks for this device or not based on wheather the value is True or False.

            driver:
                instruments_setup:
                    device_no: str - "single" or "multiple" - This depends on the number of devices being used.
                                      If the "multiple" option is used, specify the trigger source only once.

                    activation_instrument: str - () eg. "cDAQ1Mod3"  ) - The activation instrument for this device.
                                                                        Different materials can be used as dopant or host and the number of electrodes can vary.
                                                                        Once we choose input and readout electrodes, the device can be activated by applying voltages to the remaining electrodes, which we call activation electrodes.
                                                                        By tuning the voltages applied to some of the electrodes, the output current can be controlled as a function of the voltages at the remaining electrodes.
                                                                        Range - 1.2 to 0.6V or -0.7 to 0.3V​
                                                                        They have P-n junction forward bias​.Forward bias occurs when a voltage is applied such that the electric field formed by the P-N junction is decreased.
                                                                        It it is outside the range, there are Noisy solutions which are defined in the noise.py class.

                    activation_channels: list - ( eg. [8,10,13,11,7,12,14] ) - Channels through which voltages will be sent for activating the device (with both data inputs and control voltages)

                    min_activation_voltages: list - ( eg. [-1.2, -1.2, -1.2, -1.2, -1.2, -0.7, -0.7] ) - The minimum value for each of the activation electrodes expressed as a list of integers.

                    max_activation_voltages: list - ( eg. [0.6, 0.6, 0.6, 0.6, 0.6, 0.3, 0.3] ) - The maximum value for each of the activation electrodes expressed as a list of integers.

                    readout_instrument: str - (eg. "cDAQ1Mod4" )- The readout instrument for this device. It is used to accept the signal transmitted from the device.

                    readout_channels: list -( eg. [2] ) - The channels for reading the output current values expressed as a list.
                                                          The range of the readout channels depends on setup​ and on the feedback resistance produced.It also has clipping ranges​ which can be set according to preference.
                                                          Example ranges -400 to 400 or -100 to 100 nA​.

                    trigger_source: str - (eg. "cDAQ1/segment1" ) - The name of the trigger source for this device.
                                                                    The trigger source setting of the instrument determines which trigger signals are used to trigger the instrument.
                                                                    The trigger source can be set to a single channel or to any combination of channels or other trigger sources.
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
