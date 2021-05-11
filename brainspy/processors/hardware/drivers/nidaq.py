import numpy as np
import warnings
from brainspy.processors.hardware.drivers.ni.setup import (
    NationalInstrumentsSetup,
    SYNCHRONISATION_VALUE,
    CDAQ_TO_NIDAQ_RAMPING_TIME_SECONDS,
)


class CDAQtoNiDAQ(NationalInstrumentsSetup):
    """
    Class to establish a connection (for a single, or multiple hardware DNPUs) with the CDAQtoNiDAQ national instrument. It requires an additional channel to send a spike
    from the CDAQ to the NIDAQ. The data is offsetted to let the NIDAQ read the spike and start synchronising.
    """

    def __init__(self, configs):
        """
        Initialize the hardware processor. No trigger source required for this device.

            Parameters
            ----------
            configs : dict
            key-value pairs required in the configs dictionary to initialise the driver are as follows:

            real_time_rack : boolean - Only to be used when having a rack that works with real-time. True will attempt a connection to a server on the real time rack via Pyro. False will execute the drivers locally.
            sampling_frequency: int - The average number of samples to be obtained in one second, when transforming the signal from analogue to digital.
            output_clipping_range: [float,float] - The the setups have a limit in the range they can read. They typically clip at approximately +-4 V.
                Note that in order to calculate the clipping_range, it needs to be multiplied by the amplification value of the setup. (e.g., in the Brains setup the amplification is 28.5,
                is the clipping_value is +-4 (V), therefore, the clipping value should be +-4 * 28.5, which is [-110,110] (nA) ).
                The original clipping value of the surrogate models is obtained when running the preprocessing of the data in
                bspysmg.measurement.processing.postprocessing.post_process.
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
            instruments_setup:
                multiple_devices: boolean - False will initialise the drivers to read from a single hardware DNPU.
                                            True, will enable to read from more than one DNPU device at the same time.
                activation_instrument: str - Name of the activation instrument as observed in the NI Max software. E.g.,  cDAQ1Mod3
                activation_channels: list - Channels through which voltages will be sent for activating the device
                                            (both data inputs and control voltage electrodes). The channels can be
                                            checked in the schematic of the DNPU device.
                                            E.g., [8,10,13,11,7,12,14]
                activation_voltage_ranges: list - Minimum and maximum voltage for the activation electrodes. E.g., [[-1.2, 0.6], [-1.2, 0.6],
                                                    [-1.2, 0.6], [-1.2, 0.6], [-1.2, 0.6], [-0.7, 0.3], [-0.7, 0.3]]
                readout_instrument: str - Name of the readout instrument as observed in the NI Max software. E.g., cDAQ1Mod4
                readout_channels: [2] list - Channels for reading the output current values. The channels can be checked in the schematic of the DNPU device.
                trigger_source: str - For synchronisation purposes, sending data for the activation voltages on one NI Task can trigger the readout device 
                                        of another NI Task. In these cases, the trigger source name should be specified in the configs. This is only applicable for CDAQ to CDAQ setups (with or without real-time rack).
                                        E.g., cDAQ1/segment1 - More information at https://nidaqmx-python.readthedocs.io/en/latest/start_trigger.html

 -------------------------------------------------------------------------------------------------------------------------------------------------
            Appart from these values, there are some internal keys that are added internally during the initialisation of the drivers. 
            These are not required to be passed on the configs.

            offset : int - Value (in milliseconds) that the original activation voltage will be displaced,
                           in order to enable the spiking signal to reach the nidaq setup. It will be defined by SYNCHRONISATION_VALUE * sampling_frequency.
            auto_start : bool - If the task is not explicitly started with the DAQmx start_task method, it will start it anyway.
                                This value is set to False for this setup.
            max_ramping_time_seconds : int - To set the ramp time for the setup. It is defined with the flags CDAQ_TO_NIDAQ_RAMPING_TIME_SECONDS in
                                        brainspy/processors/hardware/drivers/ni/setup.py

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
        return read_data[:-1, cut_value: self.configs["data"]["shape"] + cut_value]