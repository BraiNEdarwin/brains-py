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
        c            configs : dict
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

            offset : int - Only for CDAQ TO NIDAQ setup. Value (in milliseconds) that the original activation voltage will be displaced,
                           in order to enable the spiking signal to reach the nidaq setup. The offset value is set to 1 for this setup.
            auto_start : bool - If the task is not explicitly started with the DAQmx start_task method, it will start it anyway.
                                This value is set to True for this setup.
            max_ramping_time_seconds : int - To set the ramp time for the setup. It is defined with the flags CDAQ_TO_CDAQ_RAMPING_TIME_SECONDS in
                                        brainspy/processors/hardware/drivers/ni/setup.py                                                                    The trigger source can be set to a single channel or to any combination of channels or other trigger sources.
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
