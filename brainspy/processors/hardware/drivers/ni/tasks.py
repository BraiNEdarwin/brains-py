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
import os
import time
import Pyro4
import numpy as np

import nidaqmx
import nidaqmx.constants as constants
import nidaqmx.system.device as device

from datetime import datetime
from brainspy.processors.hardware.drivers.ni.channels import init_channel_data

DEFAULT_IP = "192.168.1.5"
DEFAULT_SUBNET_MASK = "255.255.255.0"
DEFAULT_PORT = 8081

SWITCH_ETHERNET_OFF_COMMAND = "ifconfig eth0 down"
SWITCH_ETHERNET_ON_COMMAND = "ifconfig eth0 up"
RANGE_MARGIN = 0.01


def get_tasks_driver(configs):
    """

    Get a tasks driver. The tasks drivers handle all required nidaqmx.Tasks for brains-py. There
    are two task driver types. The first is the LocalTasks, that is initialised on the same
    computer where it is executed. The second is RemoteTasks, this type of driver is used for
    real-time National Instruments rack. A RemoteTasksServer is initialised on the real-time rack,
    which handles an instance of a LocalTasks driver. The RemoteTaks driver, running on the setup
    computer will connect to the RemoteTasksServer. With regard to callable methods from the
    outside, both LocalServer and RemoteServer behave seamlessly. Internally, the only difference
    is that the RemoteServer establishes a network connection with RemoteTasksServer.

    Parameters
    ----------
    configs : dict
        Configurations of the device model.
        The following keys are expected:

            real_time_rack: boolean
                It specifies whether if the National Instruments rack supports real time or not.
                A True value will instantiate a remote task driver. A False value will instantiate
                a local driver.

            uri: str
                Uniform resource identifier (URI) of the remote pyro object. This URI string can be
                obtained after runnning the method deploy_driver or run_server from tasks.py. This
                method should be run inside the real time rack, and it will prompt the URI code on
                the terminal where it is run. Additionally it will create a 'uri.txt' file in the
                same folder where it is executed.

                An example of the URI that should be copied is as follows:
                PYRO:obj_86956fef1d784982a9e2f86fec2a4fe7

                More information about it on the header of this file.

    Returns
    -------
    Union[RemoteTasks,LocalTasks] - An instance of a tasks driver.
    """
    if configs["real_time_rack"]:
        return RemoteTasks(configs["uri"])
    else:
        return LocalTasks()


def run_server(configs):
    """
    Starts a RemoteTasksServer object, which is a PYRO server. A RemoteTasksServer is meant to be
    initialised on the real-time rack, which handles an instance of a LocalTasks driver. The
    RemoteTaks driver, running on the setup computer will connect to the RemoteTasksServer. With
    regard to callable methods from the outside, both LocalServer and RemoteServer behave
    seamlessly. Internally, the only difference is that the RemoteServer establishes a network
    connection with RemoteTasksServer.

    The method assigns a static Ip addrress to the server which will be used to run Remote Tasks
    Server. Therefore, once the server is started, all method calls can be made remotely.

    Parameters
    ----------
    configs : dict
        configurations of the device model

    """
    if configs["force_static_ip"]:
        set_static_ip(configs["server"])
    server = RemoteTasksServer(configs)
    server.start()


def set_static_ip(configs):
    """
    This method is meant to be run inside the real-time rack. It gives a static IP address to the
    real-time rack, so that it always has the same address when connecting it to the main setup
    computer via Ethernet cable.

    Setting an IP can also be done manually. Check the header of this file for more information.

    IP addresses are either configured by a DHCP server or manually configured (static IP
    addresses). The subnet mask splits the IP address into the host and network addresses,
    thereby defining which part of the IP address belongs to the device and which part belongs to
    the network.

    Parameters
    ----------
    configs : dict
        configurations of the device model

    """
    SET_STATIC_IP_COMMAND = (
        f"ifconfig eth0 {configs['ip']} netmask {configs['subnet_mask']} up")
    os.system(SET_STATIC_IP_COMMAND)
    time.sleep(1)
    os.system(SWITCH_ETHERNET_OFF_COMMAND)
    time.sleep(1)
    os.system(SWITCH_ETHERNET_ON_COMMAND)


def deploy_driver(configs):
    """
    Deploy the driver based on the default IP addres, port and subnet mask specified in this file
    flags: DEFAULT_IP, DEFAULT_PORT, DEFAULT_SUBNET_MASK. This method is meant to be run inside the
    real-time rack. It will initialise an instance of RemoteTasksServer, which will enable to access
    remotely and instance of LocalTasks that will be running inside the real-time rack. Running this
    method is required in order to run an instance of RemoteTasks on the setup computer connected
    to a real-time rack.

    The remote server runs on this default IP address. An IP address is used in order to uniquely
    identify a device on an IP network. The address is divided into a network portion and host
    portion with the help of a subnet mask. A port number is always associated with an IP address.
    There ar multiple ports and we use the 8081 port (can be changed).

    Parameters
    ----------
    configs : dict
        configs of the remote server - ip,port number and subnet mask
    """
    configs["ip"] = DEFAULT_IP
    configs["port"] = DEFAULT_PORT
    configs["subnet_mask"] = DEFAULT_SUBNET_MASK
    configs["force_static_ip"] = False

    run_server(configs)


@Pyro4.expose
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

    @Pyro4.oneway
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

    @Pyro4.oneway
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

    @Pyro4.oneway
    def set_sampling_clocks(self,
                            sampling_frequency,
                            activation_clk_source,
                            samps_per_chan=1000):
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
        """
        self.activation_task.timing.cfg_samp_clk_timing(
            sampling_frequency,
            source="/" + activation_clk_source + "/ai/SampleClock",
            sample_mode=self.acquisition_type,
            samps_per_chan=samps_per_chan)
        self.readout_task.timing.cfg_samp_clk_timing(
            sampling_frequency,
            sample_mode=self.acquisition_type,
            samps_per_chan=samps_per_chan)

    @Pyro4.oneway
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

    # Main difference between read and remote read is in the try/catch
    def remote_read(self, number_of_samples_per_channel, timeout):
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

        This method also handles any DaQError that might arise from reading.

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
        try:
            return self.readout_task.read(
                number_of_samples_per_channel=number_of_samples_per_channel,
                timeout=timeout)
        except nidaqmx.errors.DaqError as e:
            print("Error reading: " + str(e))
        return -1

    @Pyro4.oneway
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

    # Main difference between write and remote write is in the try/catch
    @Pyro4.oneway
    def remote_write(self, y, auto_start):
        """
        This method writes samples to the task or virtual channels specified in the tasks of the
        tasks driver. It supports automatic start of tasks. This write method is dynamic, and is
        capable of accepting the samples to write in the various forms for most operations:

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
            Contains the samples to be written to the activation task. The shape of the samples
            should be in the shape of (activation_channel_no, sample_no).

        auto_start : Bool
            True to enable auto-start from nidaqmx drivers. False to
            start the tasks immediately after writing.        """
        self.activation_task.write(np.asarray(y), auto_start=auto_start)
        if not auto_start:
            self.activation_task.start()
            self.readout_task.start()

    @Pyro4.oneway
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

    @Pyro4.oneway
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
        self.set_sampling_clocks(
            self.configs["sampling_frequency"],
            self.configs['instruments_setup']['trigger_source'])
        return self.voltage_ranges.tolist()

    @Pyro4.oneway
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


class RemoteTasks:
    """
    This class is intended to be used only for National Instruments real-time racks. In order
    to work, it requires to have instantiated a RemoteTasksServer class on the real-time rack.
    This RemoteTasksServer class will manage an instance of a LocalTasks. The RemoteTasks simply
    declares a Pyro connection through the network via an URI. The RemoteTasks objects is just
    a wrapper around the Pyro connection which acts seamlessly in comparison with the LocalTaks,
    although internally it sends the data to the real-time rack through the network.

    """
    def __init__(self, uri):
        """
        Sets up the proxy server so that remote method calls can be made to the device. This
        enables calling methods on the Pyro object - LocalTasks.

        Parameters
        ----------
        uri : str
            uri of the remote server
        """
        self.acquisition_type = constants.AcquisitionType.FINITE
        self.tasks = Pyro4.Proxy(uri)

    def init_activation_channels(self, channel_names, voltage_ranges=None):
        """
        Wrapper for LocalTasks.init_activation_channels. More information
        can be found in that method.

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
        self.tasks.init_activation_channels(channel_names, voltage_ranges)

    def init_readout_channels(self, readout_channels):
        """
        Wrapper for LocalTasks.init_readout_channels. More information
        can be found in that method.

        Parameters
        ----------
        readout_channels : list[str]
            List containing all the readout channels of the device.
        """
        self.tasks.init_readout_channels(readout_channels)

    def add_synchronisation_channels(
        self,
        readout_instrument,
        activation_instrument,
        activation_channel_no=7,
        readout_channel_no=7,
    ):
        """
        Wrapper for LocalTasks.add_synchronisation_channels. More information
        can be found in that method.

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
        self.tasks.add_synchronisation_channels(
            readout_instrument,
            activation_instrument,
            activation_channel_no,
            readout_channel_no,
        )

    def read(self, number_of_samples_per_channel, timeout):
        """
        Wrapper for LocalTasks.remote_read. More information
        can be found in that method.

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
        return self.tasks.remote_read(
            number_of_samples_per_channel=number_of_samples_per_channel,
            timeout=timeout)

    def start_trigger(self, trigger_source):
        """
        Wrapper for LocalTasks.start_trigger. More information
        can be found in that method.

        Parameters
        ----------
        trigger_source: str
            Source trigger name. It can be
            found on the NI max program. Click on the device rack inside devices and interfaces,
            and then click on the Device Routes tab.
        """
        self.tasks.start_trigger(trigger_source)

    def write(self, y, auto_start):
        """
        Wrapper for LocalTasks.remote_write. More information
        can be found in that method.


        Parameters
        ----------
        y : np.array
            Contains the samples to be written to the activation task. The shape of the samples
            should be in the shape of (activation_channel_no, sample_no).

        auto_start : Bool
            True to enable auto-start from nidaqmx drivers. False to
            start the tasks immediately after writing.
        """
        self.tasks.remote_write(y.tolist(), auto_start)

    def stop_tasks(self):
        """
        Wrapper for LocalTasks.stop_tasks. More information
        can be found in that method.
        """
        self.tasks.stop_tasks()

    def init_tasks(self, configs):
        """
        Wrapper for LocalTasks.init_tasks. More information
        can be found in that method.

        Parameters
        ----------
        configs : dict
            Configs dictionary for the device model

        Returns
        -------
        list
            list of voltage ranges
        """
        self.voltage_ranges = np.array(self.tasks.init_tasks(configs))

    def close_tasks(self):
        """
        Wrapper for LocalTasks.close_tasks. More information
        can be found in that method.
        """
        self.tasks.close_tasks()


class RemoteTasksServer:
    """
    Server for the Remote Tasks on the device.
    To be able to call methods on a Pyro object , an appropriate URI is created and given to the
    server. The class uses the Pyro remote server. It enables objects between LocalTasks and
    RemoteTasks to talk to each other over the network.

    This class provides the object -  (LocalTasks object) and actually runs the methods.

    """
    def __init__(self, configs):
        """
        Initialize the remote server based on the configurations of the model and initialize the
        LocalTasks. Remote method calls can be made after the server is initialized.

        Parameters
        ----------
        configs : dict
            configs of the model
        """
        self.configs = configs
        self.tasks = LocalTasks()

    def save_uri(self, uri):
        """
        Stores the Uniform resource identifier (URI) of the remote server that hosts the LocalTasks
        Pyro object to a text file.

        Parameters
        ----------
        uri : str
            Uniform resource identifier (URI) of the remote pyro object. This URI string can be
            obtained after registering the daemon of the server.

            An example of an URI is as follows:
            PYRO:obj_86956fef1d784982a9e2f86fec2a4fe7

        """
        print("Server ready, object URI: " + str(uri))
        f = open("uri.txt", "w")
        f.write(str(uri) + " \n")
        f.close()

    def start(self):
        """
        Starts a daemon thread to handle to the remote server and the remote tasks on the device.
        A thread is a separate flow of execution. It registers an instance of LocalTasks on Pyro.
        This method starts a daemon thread which will shut down immediately when the program exits.

        """
        self.daemon = Pyro4.Daemon(host=self.configs["ip"],
                                   port=self.configs["port"])
        uri = self.daemon.register(self.tasks)
        self.save_uri(uri)
        self.daemon.requestLoop()

    def stop(self):
        """
        Method to stop the remote server by closing the daemon thread.
        """
        self.daemon.close()


if __name__ == "__main__":
    deploy_driver({})
