"""
This file contains drivers to handle nidaqmx.Tasks on different environments, that include regular
National Instrument racks or National Instrument real-time racks.

Both drivers work seamlessly, they declare the following nidaqmx.Task instances:
        * activation_task: It handles sending signals to the (DNPU) device through electrodes
                           declared as activation electrodes.
        * readout_task: It handles reading signlas comming out from the (DNPU) device from
                        electrodes declared as readout electrodes.

Both nidaqmx.Task instances will declare a channel per electrode, and in the case of the cdaq
to nidaq connection, they will also declare an extra synchronization channel.
It can alo be used to set the shape variables according to the requiremnets.
"""
import sys
import nidaqmx
import nidaqmx.constants as constants
import nidaqmx.system.device as device

import numpy as np
from datetime import datetime
from brainspy.processors.hardware.drivers.ni.channels import init_channel_data

RANGE_MARGIN = 0.01


class IOTasksManager:
    """
    Class to initialise and handle the "nidaqmx.Task"s required for brains-py drivers.

    More information about NI tasks can be found at:
    https://nidaqmx-python.readthedocs.io/en/latest/task.html

    """
    def __init__(self, configs):
        """
        It declares the following nidaqmx.Task instances:
            * activation_task: It handles sending signals to the (DNPU) device through electrodes
                            declared as activation electrodes.
            * readout_task: It handles reading signlas comming out from the (DNPU) device from
                            electrodes declared as readout electrodes.

        Both nidaqmx.Task instances will declare a channel per electrode, and in the case of the
        cdaq to nidaq connection, they will also declare an extra synchronization channel.
        It can alo be used to set the shape variables according to the requiremnets.
        These tasks will be updated as the method calls are made to do different types of tasks.
        It also initializes the device to Acquire or generate a finite number of samples
        """
        assert type(configs) == dict, "The configs should be of type - dict"
        self.acquisition_type = constants.AcquisitionType.FINITE
        self.activation_task = None
        self.readout_task = None
        try:
            self.init_tasks(configs)
        except Exception as e:
            if not 'devices' in dir(self):
                self.devices = []
            self.close_tasks()
            raise e

    def init_activation_channels(self, channel_names, voltage_ranges=None):
        """
        Initialises the activation channels connected to the activation electrodes of the device.
        These are being sent as list of voltage values and channel names which are the start values
        for device.

        Parameters
        ----------
        channel_names : list
            List of the names of the activation channels

        voltage_ranges : Optional[list] or numpy.ndarray
            List/numpy.ndarray of maximum and minimum voltage ranges that will be allowed to be sent through each
            channel. The  dimension of the list should be (channel_no,2) where the second dimension
            stands for min and max values of the range, respectively. When set to None, there will
            be no specific limitations on what can be sent through the device. This could
            potentially cause damages to the device. By default is None.
        """
        assert type(channel_names
                    ) == list, "The channel_names should be of type - list"
        if voltage_ranges is not None:
            assert type(voltage_ranges) == list or type(
                voltage_ranges
            ) == np.ndarray, "The voltage_ranges should be of type - list or numpy array"
            assert len(channel_names) == len(
                voltage_ranges
            ), "The length of channel_names should be equal to the length of voltage ranges"
            for voltage_range in voltage_ranges:
                assert type(voltage_range) == list or type(
                    voltage_range
                ) == np.ndarray, "Each voltage range should be a list of 2 values"
                assert len(
                    voltage_range
                ) == 2, "Voltage range should contain 2 values : max and min"
                assert isinstance(
                    voltage_range[0], (np.floating, float, int)
                ), "Volatge range can contain only int or float type values"
                assert isinstance(
                    voltage_range[1], (np.floating, float, int)
                ), "Volatge range can contain only int or float type values"

        self.activation_task = nidaqmx.Task(
            "activation_task_" +
            datetime.utcnow().strftime("%Y_%m_%d_%H%M%S_%f"))
        for i in range(len(channel_names)):
            channel_name = str(channel_names[i])
            if voltage_ranges is not None:
                assert (
                    voltage_ranges[i][0] > -2 and voltage_ranges[i][0] < 2
                ), "Minimum voltage ranges configuration outside of the allowed values -2 and 2"
                assert (
                    voltage_ranges[i][1] > -2 and voltage_ranges[i][1] < 2
                ), "Maximum voltage ranges configuration outside of the allowed values -2 and 2"
                self.activation_task.ao_channels.add_ao_voltage_chan(
                    channel_name,
                    min_val=voltage_ranges[i][0].item() - RANGE_MARGIN,
                    max_val=voltage_ranges[i][1].item() + RANGE_MARGIN,
                )
            else:
                print(
                    "WARNING! READ CAREFULLY THIS MESSAGE. Activation channels have been"
                    +
                    "initialised without a security voltage range, they will be automatically set"
                    +
                    "up to a range between -2 and 2 V. This may result in damaging the device."
                    +
                    "Press ENTER only if you are sure that you want to proceed, otherwise STOP "
                    +
                    "the program. Voltage ranges can be defined in the instruments setup"
                    + "configurations.")
                input()
                self.activation_task.ao_channels.add_ao_voltage_chan(
                    channel_name,
                    min_val=-2,
                    max_val=2,
                )

    def init_readout_channels(self, readout_channels):
        """
        Initializes the readout channels corresponding to the readout electrodes of the device.
        The range of the readout channels depends on setup​ and on the feedback resistance produced.

        Parameters
        ----------
        readout_channels : list[str]
            List containing all the readout channels of the device.
        """
        assert type(readout_channels
                    ) == list, "The readout_channels should be of type - list"
        for readout_channel in readout_channels:
            assert type(
                readout_channel
            ) == str, "Each readout_channel should be of type - str"
        self.readout_task = nidaqmx.Task(
            "readout_task_" + datetime.utcnow().strftime("%Y_%m_%d_%H%M%S_%f"))
        for i in range(len(readout_channels)):
            channel = readout_channels[i]
            self.readout_task.ai_channels.add_ai_voltage_chan(channel)

    def set_sampling_frequencies(self, activation_sampling_frequency: int,
                                 readout_sampling_frequency: int,
                                 points_to_write: int, points_to_read: int):
        """
        Sets the sampling frequency for the activation and readout tasks. The activation
        task is in charge to send signals from the  computer to the NI module, and the
        readout task is in charge of reading signals from the NI module.
        Parameters
        ----------
        activation_sampling_frequency : int
            The number of samples that the activation task will obtain in one second.
        readout_sampling_frequency : int
            The number of samples that the readout task will obtain in one second.
        points_to_write : int
            Number of points that will be written.
        points_to_read: int
            Number of points that are expected to be read given the number
            of points to be written, and the activation and readout frequencies.
        """
        assert type(
            activation_sampling_frequency
        ) == int, "The activation_sampling_frequency value should be of type - int"
        assert type(
            readout_sampling_frequency
        ) == int, "The readout_sampling_frequency value should be of type - int"
        assert type(
            points_to_write
        ) == int, "The points_to_write value should be of type - int"
        assert type(
            points_to_read
        ) == int, "The points_to_read value should be of type - int"

        self.activation_task.timing.cfg_samp_clk_timing(
            activation_sampling_frequency,
            sample_mode=self.acquisition_type,
            samps_per_chan=points_to_write,
        )
        self.readout_task.timing.cfg_samp_clk_timing(
            readout_sampling_frequency,
            sample_mode=self.acquisition_type,
            samps_per_chan=points_to_read,
        )

    def add_synchronisation_channels(
        self,
        readout_instrument,
        activation_instrument,
        activation_channel_no=7,
        readout_channel_no=7,
    ):
        """
        The method is used to add a synchronized activation and readout channel to the device
        when using cdaq to nidaq devices. Activation channels send signals to the activation
        electrodes while an additional synchronisation channel is created to communicate with
        the nidaq readout module. A spike is sent through this synchronisation channel to make
        the nidaq and the cdaq write and read syncrhonously.

        Parameters
        ----------
        readout_instrument: str
            Name of the instrument from which the read will be performed on the readout electrodes.
        activation_instrument : str
            Name of the instrument writing signals to the activation electrodes.
        activation_channel_no : int, optional
            Channel through which voltages will be sent for activating the device (with both data
            inputs and control voltages), by default 7.
        readout_channel_no : int, optional
            Channel for reading the output current values, by default 7.
        """
        assert type(readout_instrument
                    ) == str, "The readout_instrument should be of type - str"
        assert type(
            activation_instrument
        ) == str, "The actication_instrument should be of type - str"
        assert type(
            activation_channel_no
        ) == int, "The activation_channel_no should be of type - int"
        assert type(readout_channel_no
                    ) == int, "The readout_channel_no should be of type - int"

        # Define ao7 as sync signal for the NI 6216 ai0
        self.activation_task.ao_channels.add_ao_voltage_chan(
            activation_instrument + "/ao" + str(activation_channel_no),
            name_to_assign_to_channel="activation_synchronisation_channel",
            min_val=-5,
            max_val=5,
        )
        self.readout_task.ai_channels.add_ai_voltage_chan(
            readout_instrument + "/ai" + str(readout_channel_no),
            name_to_assign_to_channel="readout_synchronisation_channel",
            min_val=-5,
            max_val=5,
        )

    def read(self, number_of_samples_per_channel, timeout=None):
        """
        Reads samples from the task or virtual channels you specify. This read method is dynamic,
        and is capable of inferring an appropriate return type based on these factors: - The
        channel type of the task. - The number of channels to read. - The number of samples per
        channel. The data type of the samples returned is independently determined by the channel
        type of the task. For digital input measurements, the data type of the samples returned is
        determined by the line grouping format of the digital lines. If the line grouping format is
        set to “one channel for all lines”, the data type of the samples returned is int. If the
        line grouping format is set to “one channel per line”, the data type of the samples
        returned is boolean. If you do not set the number of samples per channel, this method
        assumes one sample was requested. This method then returns either a scalar (1 channel to
        read) or a list (N channels to read).

        If you set the number of samples per channel to ANY value (even 1), this method assumes
        multiple samples were requested. This method then returns either a list (1 channel to read)
        or a list of lists (N channels to read).

        Original documentation from: https://nidaqmx-python.readthedocs.io/en/latest/task.html

        Parameters
        ----------
        number_of_samples_per_channel : int
            Specifies the number of samples to read. If this input is not set, assumes samples to
            read is 1. Conversely, if this input is set, assumes there are multiple samples to read.
            If you set this input to nidaqmx.constants. READ_ALL_AVAILABLE, NI-DAQmx determines how
            many samples to read based on if the task acquires samples continuously or acquires a
            finite number of samples. If the task acquires samples continuously and you set this
            input to nidaqmx.constants.READ_ALL_AVAILABLE, this method reads all the samples
            currently available in the buffer. If the task acquires a finite number of samples and
            you set this input to nidaqmx.constants.READ_ALL_AVAILABLE, the method waits for the
            task to acquire all requested samples, then reads those samples. If you set the
            “read_all_avail_samp” property to True, the method reads the samples currently
            available in the buffer and does not wait for the task to acquire all requested samples.

        timeout : Optional[float]
            Specifies the amount of time in seconds to wait for samples to become available.
            If the time elapses, the method returns an error and any samples read before the
            timeout elapsed. The default timeout is 10 seconds. If you set timeout to
            nidaqmx.constants.WAIT_INFINITELY, the method waits indefinitely. If you set timeout to
            0, the method tries once to read the requested samples and returns an error if it is
            unable to.

        Returns
        -------
        list
            The samples requested in the form of a scalar, a list, or a list of lists. See method
            docstring for more info. NI-DAQmx scales the data to the units of the measurement,
            including any custom scaling you apply to the channels.  Use a DAQmx Create Channel
            method to specify these units.
        """
        if number_of_samples_per_channel is not None:
            assert (
                type(number_of_samples_per_channel)
            ) == int, "number_of_samples_per_channel should be of type - int"
            assert (number_of_samples_per_channel > 0
                    ), "number_of_samples_per_channel value cannot be negative"
        if timeout is not None:
            assert type(timeout) == float or type(
                timeout) == int, "timeout should be of type float or int"
            assert timeout >= 0, "timeout value cannot be negative"

        return self.readout_task.read(
            number_of_samples_per_channel=number_of_samples_per_channel,
            timeout=timeout)

    def start_trigger(self, trigger_source):
        """
        To synchronise cdaq to cdaq modules a start trigger can be set,
        in such a way that when a write operation is done, a read operation
        is triggered.

        More information can be found in:
        https://nidaqmx-python.readthedocs.io/en/latest/start_trigger.html

        Parameters
        ----------
        trigger_source: str
            Source trigger name. It can be
            found on the NI max program. Click on the device rack inside devices and interfaces,
            and then click on the Device Routes tab.
        """
        assert type(
            trigger_source
        ) == str, "The name of the trigger source should be of type - str"
        self.activation_task.triggers.start_trigger.cfg_dig_edge_start_trig(
            "/" + trigger_source + "/ai/StartTrigger")

    def write(self, y, auto_start):
        """
        This method writes samples to the task or virtual channels specified in the tasks of the
        tasks driver. It also catches DaqError exceptions. It supports automatic start of tasks.
        This write method is dynamic, and is capable of accepting the samples to write in the
        various forms for most operations:

            Scalar: Single sample for 1 channel.
            List/1D numpy.ndarray: Multiple samples for 1 channel or 1 sample for multiple channels.
            List of lists/2D numpy.ndarray: Multiple samples for multiple channels.

        The data type of the samples passed in must be appropriate for the channel type of the task.

        For counter output pulse operations, this write method only accepts samples in these forms:

            Scalar CtrFreq, CtrTime, CtrTick (from nidaqmx.types): Single sample for 1 channel.
            List of CtrFreq, CtrTime, CtrTick (from nidaqmx.types): Multiple samples for 1 channel
            or 1 sample for multiple channels.

        If the task uses on-demand timing, this method returns only after the device generates all
        samples. On-demand is the default timing type if you do not use the timing property on the
        task to configure a sample timing type. If the task uses any timing type other than on-
        demand, this method returns immediately and does not wait for the device to generate all
        samples. Your application must determine if the task is done to ensure that the device
        generated all samples.

        Both of these tasks occur simulataneously and are synchronized.
        The tasks will start automatically if the "auto_start" option is set to True in the
        configuration dictionary used to intialize this device.

        Parameters
        ----------
        y : np.array
            Contains the samples to be written to the activation task.

        auto_start : Bool
            True to enable auto-start from nidaqmx drivers. False to
            start the tasks immediately after writing.        """

        assert type(
            y
        ) == np.ndarray, "The sample data: y should be of type - numpy array"
        assert type(
            auto_start) == bool, "auto_start param should be of type - bool"

        y = np.require(y, dtype=y.dtype, requirements=["C", "W"])
        try:
            self.activation_task.write(y, auto_start=auto_start)
            if not auto_start:
                self.activation_task.start()
                self.readout_task.start()
        except nidaqmx.errors.DaqError as error:
            print("There was an error writing to the activation task: " +
                  self.activation_task.name + "\n" + str(error))
            self.close_tasks()
            sys.exit(1)

    def stop_tasks(self):
        """
        To stop the all tasks on this device namely - the activation tasks and the readout tasks to
        and from the device.
        """
        self.readout_task.stop()
        self.activation_task.stop()

    def init_tasks(self, configs):
        """
        To Initialize the tasks on the device based on the configurations dictionary provided.
        the method initializes the activation and readout tasks from the device by setting the
        voltage ranges and choosing the instruments that have been specified.

        Parameters
        ----------
        configs : dict
            Configs dictionary for the device model

            The configs should have the following keys:

                sampling_frequency: int
                    The average number of samples to be obtained in one second, when transforming
                    the signal from analogue to digital.
                amplification: float
                    The output current (nA) of the device is converted by the readout hardware to
                    voltage (V), because it is easier to do the readout of the device in voltages.
                    This output signal in nA is amplified by the hardware when doing this current
                    to voltage conversion, as larger signals are easier to detect. In order to
                    obtain the real current (nA) output of the device, the conversion is
                    automatically corrected in software by multiplying by the amplification value
                    again.
                    The amplification value depends on the feedback resistance of each of the
                    setups.

                    Below, there is a guide of the amplification value needed for each of the
                    setups:

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
                                        If no correction is desired, the amplification can be set
                                        to 1.
            instruments_setup:
                multiple_devices: boolean
                    False will initialise the drivers to read from a single hardware DNPU.
                    True, will enable to read from more than one DNPU device at the same time.
                activation_instrument: str
                    Name of the activation instrument as observed in the NI Max software.
                    E.g.,  cDAQ1Mod3
                activation_channels: list
                    Channels through which voltages will be sent for activating the device
                    (both data inputs and control voltage electrodes). The channels can be
                    checked in the schematic of the DNPU device.
                    E.g., [8,10,13,11,7,12,14]
                activation_voltage_ranges: list
                    Minimum and maximum voltage for the activation electrodes.
                    E.g., [[-1.2, 0.6], [-1.2, 0.6],
                    [-1.2, 0.6], [-1.2, 0.6], [-1.2, 0.6], [-0.7, 0.3], [-0.7, 0.3]]
                readout_instrument: str
                    Name of the readout instrument as observed in the NI Max software.
                    E.g., cDAQ1Mod4
                readout_channels: [2] list
                    Channels for reading the output current values.
                    The channels can be checked in the schematic of the DNPU device.
                trigger_source: str
                    For synchronisation purposes, sending data for the activation voltages on
                    one NI Task can trigger the readout device of another NI Task. In these
                    cases,the trigger source name should be specified in the configs.
                    This is only applicable for CDAQ to CDAQ setups
                    (with or without real-time rack).
                    E.g., cDAQ1/segment1
                    More information at
                    https://nidaqmx-python.readthedocs.io/en/latest/start_trigger.html
            plateau_length: float - Length of the plateau that is being sent through the forward
            call of the HardwareProcessor
            slope_length : float - Length of the slopes in the waveforms sent to the device through
            the drivers

        Returns
        -------
        list
            list of voltage ranges
        """
        assert type(configs) == dict, "The configs should be of type - dict"
        self.configs = configs
        (
            self.activation_channel_names,
            self.readout_channel_names,
            instruments,
            self.voltage_ranges,
        ) = init_channel_data(configs)
        devices = []
        for instrument in instruments:
            devices.append(device.Device(name=instrument))
        self.devices = devices
        # TODO: add a maximum and a minimum to the activation channels
        self.init_activation_channels(self.activation_channel_names,
                                      self.voltage_ranges)
        self.init_readout_channels(self.readout_channel_names)
        return self.voltage_ranges.tolist()

    def close_tasks(self):
        """
        Close all the task on this device -  both activation and readout tasks by deleting them.
        Note - This method is different from the stop_tasks() method which only stops the current
        tasks temporarily.
        """
        if self.readout_task is not None:
            self.readout_task.close()
            del self.readout_task
            self.readout_task = None
        if self.activation_task is not None:
            self.activation_task.close()
            del self.activation_task
            self.activation_task = None

        for dev in self.devices:
            dev.reset_device()
