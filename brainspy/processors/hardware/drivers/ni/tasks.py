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

The usage of regular National Instrument racks is simple. You only have to handle an instance of
LocalTasks. The usage of the National Instrument real-time rack requires a different procedure. On
one hand, an instance of a RemoteTasksServer should be running on the real-time rack. This instance
internally declares an instance of LocalTasks, and makes them accessible via computer network (in
this case through Ethernet cable). On the other hand, the setup computer will have an instance of
RemoteTasks, that will connect to the RemoteTasksServer via ethernet using Pyro. Pyro is a library
that enables you to build applications in which objects can talk to each other over the network,
with minimal programming effort. In this way, a RemoteTask and a LocalTasks will behave seamlessly,
in terms of callable methods.

    0 - Connect the computer with the real-time rack using USB connection. After this, check if you
    can access the real-time rack via ssh (if you are on Windows you can use Putty). You can use
    the NI Device Monitor application to check the domain name of the real-time rack. It should
    look like: NI-cDAQ-9132-01DD36D9. Once inside the real-time rack, make sure you are on the right
    user account. You can create another one if necessary. Make sure it has adequate permissions.
    Make sure that the user you select is the same one for the rest of the installation.
    More information on creating users can be found at:
    https://linuxize.com/post/how-to-create-users-in-linux-using-the-useradd-command/

    1 - Make sure that the real-time-rack has connection to the internet. You can connect an
        ethernet cable directly to the internet. Once having it ready, you can install python,
        and the brains-py package, following the regular installation instructions of the wiki.
        Once you have finished, disconnect the Ethernet cable.

    2 - The computer needs to be connected to the real-time rack via ethernet cable. In order for
       them to communicate the connection needs to be established on an IP address that is on the
       same subnetwork. The subnetwork is defined by the subnet mask. More information on how IP
       works can be found in https://en.wikipedia.org/wiki/IP_address

       In order to ensure that the IPs are on the same subnetwork, the following procedures can be
       done:
        * On the real-time rack:
            You can communicate with the real time rack via ssh, as explained in step 0. Then, run
            the command: "ifconfig" and look at the eth0 ip address and check if:

            inet addr: 192.168.1.5
            mask: 255.255.255.0

            If this is not the case set it using the following command
            "ifconfig eth0 192.168.1.5 netmask 255.255.255.0 up". Alternatively,
            you can also run the set_static_ip method from this class.

            Check with ifconfig if the eth0 address has changed.
        * On the setup computer:
            If you are on linux, repeat the same procedure as above,
            but use the command "ifconfig eth0 192.168.1.5 netmask 255.255.255.0 up" instead.
            If you are using Windows, go to the properties of the ethernet connection. You can
            specify there the internet protocol version 4 (TCP/IPv4) properties.

            Make use the the use the following IP adress cicle is selected and fill in the following
            information:

            Ip adress: 	192.168.1.4
            Subenet mask:	255.255.255.0
            default gateway:Leave Blank

            Also make sure that the following DNS server adresses circle is filled and fill in:
            Prefered DNS server: 	8.8.8.8
            Alternate DNS server	Leave Blank

    3 - Disconnect the USB cable and connect the ethernet cable to the eth0 of the real-time rack,
        and the configured ethernet port on the computer. Remember that the ethernet cable needs to
        be a crossover cable and not a direct cable since you are connecting two devices directly.

    4 - Connect again via ssh, in a similar way to that explained in step 0, but using the ethernet
        cable instead of the USB one. Once inside the real-time rack you can run the method
        deploy_driver or run_server from tasks.py using python. deploy_driver can be run by
        simply runnning python tasks.py. This method should be run inside the real time rack,
        and it will prompt the URI code on the terminal where it is run. Additionally it will
        create a 'uri.txt' file in the same folder where it is executed.

        An example of the URI that should be copied is as follows:
        PYRO:obj_86956fef1d784982a9e2f86fec2a4fe7

    5 -  Now the Pyro connection should be ready. To use it, you should change the driver in the
         config file to remote and fill in the corresponding URL that is returned when running the
         previous file. If the connection is stil open and the information properly added to the
         config file you will be able to perform measurements in the same manner as on the other
         systems. If you do not restart the CDAQ computer it will stay the same allowing all the
         other steps to be done on the measurement PC if the usb_driver does not work or is not
         present.

"""
import nidaqmx
import nidaqmx.constants as constants
import nidaqmx.system.device as device

import numpy as np
from datetime import datetime
from brainspy.processors.hardware.drivers.ni.channels import init_channel_data

RANGE_MARGIN = 0.01


class LocalTasks:
    """
    Class to initialise and handle the "nidaqmx.Task"s required for brains-py drivers.
    Some methods of this class are declared with Pyro flags to make them available for remote
    access.

    @Pyro.oneway -
    For calls to such methods, Pyro will not wait for a response from the remote object.
    The return value of these calls is always None.

    More information about NI tasks can be found at:
    https://nidaqmx-python.readthedocs.io/en/latest/task.html
    More information about remote objects can be found at:
    https://pyro4.readthedocs.io/en/stable/

    """
    def __init__(self):
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
        self.acquisition_type = constants.AcquisitionType.FINITE
        self.activation_task = None
        self.readout_task = None

    def init_activation_channels(self, channel_names, voltage_ranges=None):
        """
        Initialises the activation channels connected to the activation electrodes of the device.
        These are being sent as list of voltage values and channel names which are the start values
        for device.

        Parameters
        ----------
        channel_names : list
            List of the names of the activation channels

        voltage_ranges : Optional[list]
            List of maximum and minimum voltage ranges that will be allowed to be sent through each
            channel. The  dimension of the list should be (channel_no,2) where the second dimension
            stands for min and max values of the range, respectively. When set to None, there will
            be no specific limitations on what can be sent through the device. This could
            potentially cause damages to the device. By default is None.
        """
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
        self.readout_task = nidaqmx.Task(
            "readout_task_" + datetime.utcnow().strftime("%Y_%m_%d_%H%M%S_%f"))
        for i in range(len(readout_channels)):
            channel = readout_channels[i]
            self.readout_task.ai_channels.add_ai_voltage_chan(channel)

    def set_sampling_frequencies(self, activation_sampling_frequency,
                                 readout_sampling_frequency, points_to_write):
        """
        One way method to set the shape variables for the data that is being sent to the device.
        Depending on which device is being used, CDAQ or NIDAQ, and the sampling frequency, the
        shape of the data that is being sent can to be specified.


        Parameters
        ----------
        sampling_frequency : float
            The average number of samples to be obtained in one second.
        samples_per_chan : (int,int)
            Number of expected samples, per channel.

        Returns
        -------
        points_to_read: int
            Number of points that are expected to be read given the number
            of points to be written, and the activation and readout frequencies.
        """
        points_to_read = int(
            (readout_sampling_frequency / activation_sampling_frequency) *
            points_to_write)
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
        return points_to_read

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

    def read(self, number_of_samples_per_channel, timeout):
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
        number_of_samples_per_channel : Optional[int]
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

        y = np.require(y, dtype=y.dtype, requirements=["C", "W"])
        try:
            self.activation_task.write(y, auto_start=auto_start)
        except nidaqmx.errors.DaqError:
            print("There was an error writing to the activation task: " +
                  self.activation_task.name)
            print("Trying to reset device and do the read again.")
            for dev in self.devices:
                dev.reset_device()
            self.init_activation_channels(self.activation_channel_names,
                                          self.voltage_ranges)
            self.activation_task.write(y, auto_start=auto_start)

        if not auto_start:
            self.activation_task.start()
            self.readout_task.start()

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
            configs dictionary for the device model

            The configs should have the following keys:

             processor_type : str
                "simulation_debug" or "cdaq_to_cdaq" or "cdaq_to_nidaq" - Processor type to
                initialize a hardware processor
            driver:
                real_time_rack : boolean
                    Only to be used when having a rack that works with real-time.
                    True will attempt a connection to a server on the real time rack via Pyro.
                    False will execute the drivers locally.
                sampling_frequency: int
                    The average number of samples to be obtained in one second, when transforming
                    the signal from analogue to digital.
                output_clipping_range: [float,float]
                    The the setups have a limit in the range they can read. They typically clip at
                    approximately +-4 V.
                    Note that in order to calculate the clipping_range, it needs to be multiplied
                    by the amplification value of the setup.
                    (e.g., in the Brains setup the amplification is 28.5, is the clipping_value is
                    +-4 (V), therefore,
                    the clipping value should be +-4 * 28.5, which is [-110,110] (nA) ).
                    The original clipping value of the surrogate models is obtained when running
                    the preprocessing of the data in
                    bspysmg.measurement.processing.postprocessing.post_process.
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