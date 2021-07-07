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
    Get the tasks driver based on the configurations dictionary.
    The tasks driver can either be RemoteTasks or LocalTasks.
    The Local Tasks use Pyro which are Python Remote Objects. Objects can talk to eachother over the network with minimal effort.
    You can just use normal Python method calls to call objects on other machines.

    Parameters
    ----------
    configs : dict
        configurations of the device model

    Returns
    -------
    class - RemoteTasks() or LocalTasks() based on configs defined
    """
    if configs["real_time_rack"]:
        return RemoteTasks(configs["uri"])
    else:
        return LocalTasks()


def run_server(configs):
    """
    To start and run the server for the RemoteTasks.
    The method assigns a static Ip addrress to the server which will be used to run Remote Tasks Server.
    Therefore, once the server is started, all method calls can be made remotely.

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
    Give a static IP address to the system with a mask.
    IP addresses are either configured by a DHCP server or manually configured (static IP addresses).
    The subnet mask splits the IP address into the host and network addresses,
    thereby defining which part of the IP address belongs to the device and which part belongs to the network.

    Parameters
    ----------
    configs : dict
        configurations of the device model

    """
    SET_STATIC_IP_COMMAND = (
        f"ifconfig eth0 {configs['ip']} netmask {configs['subnet_mask']} up"
    )
    os.system(SET_STATIC_IP_COMMAND)
    time.sleep(1)
    os.system(SWITCH_ETHERNET_OFF_COMMAND)
    time.sleep(1)
    os.system(SWITCH_ETHERNET_ON_COMMAND)


@Pyro4.expose
class LocalTasks:
    """
    Class to initialize and handle the Local tasks on the device.
    They include all tasks which are needed to control the device including the activation,readout and synchronization of channels.
    It can alo be used to set the shape variables according to the requiremnets.

    Methods of this class are available for remote access.
    The Local Tasks use Pyro which are Python Remote Objects.
    Objects can talk to eachother over the network with minimal effort.
    You can just use normal Python method calls to call objects on other machines.
    Pyro can be used to distribute and integrate various kinds of resources or responsibilities: computational (hardware) resources (cpu, storage),and also informational resources (data).

    @Pyro.oneway -
    For calls to such methods, Pyro will not wait for a response from the remote object.
    The return value of these calls is always None.

    More information about NI tasks can be found at: https://nidaqmx-python.readthedocs.io/en/latest/task.html
    More information about remote objects can be found at: https://pyro4.readthedocs.io/en/stable/

    """

    def __init__(self):
        """
        Initialize the activation and readout task as Null values in the beginning.
        These tasks will be updated as the method calls are made to do different types of tasks.
        It also initializes the device to Acquire or generate a finite number of samples
        """
        self.acquisition_type = constants.AcquisitionType.FINITE
        self.activation_task = None
        self.readout_task = None

    @Pyro4.oneway
    def init_activation_channels(self, channel_names, voltage_ranges=None):
        """
        Initialises the activation electrodes of the device.
        The output of the computer is the input of the device.
        These are being sent as list of voltage values and channel names which are the start values for device. They can however be tuned later according to requirements.
        Activation electrodes -
            Range - 1.2 to 0.6V or -0.7 to 0.3V​
            They have P-n junction forward bias​.Forward bias occurs when a voltage is applied such that the electric field formed by the P-N junction is decreased.
            If it is outside the range, there are Noisy solutions which are defined in the noise.py class.

        Parameters
        ----------
        channel_names : list
            list of the names of the activation channels for this device
        voltage_ranges : list, optional
            list of voltage ranges for this device, by default None
        """
        self.activation_task = nidaqmx.Task(
            "activation_task_" + datetime.utcnow().strftime("%Y_%m_%d_%H%M%S_%f")
        )
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
                    "WARNING! READ CAREFULLY THIS MESSAGE. Activation channels have been initialised without a security voltage range, they will be automatically set up to a range between -2 and 2 V. This may result in damaging the device. Press ENTER only if you are sure that you want to proceed, otherwise STOP the program. Voltage ranges can be defined in the instruments setup configurations."
                )
                input()
                self.activation_task.ao_channels.add_ao_voltage_chan(
                    channel_name,
                    min_val=-2,
                    max_val=2,
                )

    @Pyro4.oneway
    def init_readout_channels(self, readout_channels):
        """
        Initializes the readout instrument of the device.
        The input of the computer which is the output of the device
        The range of the readout channels depends on setup​ and on the feedback resistance produced.It also has clipping ranges​ which can be set according to preference.
        Example ranges -400 to 400 or -100 to 100 nA​.

        Parameters
        ----------
        readout_channels : list
            list of readout channes for the device
        """
        self.readout_task = nidaqmx.Task(
            "readout_task_" + datetime.utcnow().strftime("%Y_%m_%d_%H%M%S_%f")
        )
        for i in range(len(readout_channels)):
            channel = readout_channels[i]
            self.readout_task.ai_channels.add_ai_voltage_chan(channel)

    @Pyro4.oneway
    def set_sampling_clocks(self, sampling_frequency, activation_clk_source, samps_per_chan=1000):
        """
        One way method to set the shape variables for the data that is being sent to the device.
        Depending on which device is being used, CDAQ or NIDAQ, and the sampling frequency, the shape of the data that is being sent can to be specified.
        This function helps to tackle the problem of differnt batches having differnt data shapes (for example - differnt sample size) when dealing with big data.

        Parameters
        ----------
        sampling_frequency : float
             the average number of samples to be obtained in one second
        shape : (int,int)
            required shape of for sampling
        """
        self.activation_task.timing.cfg_samp_clk_timing(
            sampling_frequency,
            source="/" + activation_clk_source + "/ai/SampleClock",
            sample_mode=self.acquisition_type,
            samps_per_chan=samps_per_chan
        )
        self.readout_task.timing.cfg_samp_clk_timing(
            sampling_frequency,
            sample_mode=self.acquisition_type,
            samps_per_chan=samps_per_chan
        )

#    @Pyro4.oneway
#    def set_shape(self, sampling_frequency, shape):
#        """
#        One way method to set the shape variables for the data that is being sent to the device.
#        Depending on which device is being used, CDAQ or NIDAQ, and the sampling frequency, the shape of the data that is being sent can to be specified.
#        This function helps to tackle the problem of differnt batches having differnt data shapes (for example - differnt sample size) when dealing with big data.
#
#        Parameters
#        ----------
#        sampling_frequency : float
#             the average number of samples to be obtained in one second
#        shape : (int,int)
#            required shape of for sampling
#        """
#        self.activation_task.timing.cfg_samp_clk_timing(
#            sampling_frequency,
#            sample_mode=self.acquisition_type,
#            samps_per_chan=shape,
#        )
#        self.readout_task.timing.cfg_samp_clk_timing(
#            sampling_frequency,
#            sample_mode=self.acquisition_type,
#            samps_per_chan=shape,  # TODO: Add shape + 1 ?
#        )

    @Pyro4.oneway
    def add_synchronisation_channels(
        self,
        readout_instrument,
        activation_instrument,
        activation_channel_no=7,
        readout_channel_no=7,
    ):
        """
        The method is used to add a synchronized activation and readout channel to the device.
        Channels can be used to synchronize goroutines which are read/write tasks on the device.
        A channel can make a goroutine wait until its finished. Sometimes a goroutine needs to be finished before you can start the next one (synchronous). This can be solved with channels.

        Parameters
        ----------
        readout_instrument: str
            name of the readout instrument
        activation_instrument : str
            name of the activation instrument
        activation_channel_no : int, optional
            Channel through which voltages will be sent for activating the device (with both data inputs and control voltages), by default 7
        readout_channel_no : int, optional
            Channel for reading the output current values, by default 7
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

    def read(self, offsetted_shape, ceil):
        """
        Read from the readout channel but with an offset value which can be specified. Th function alos takes a ceil value as a parameter, therefore, all reads are under this maximum value.
        Offset on the read is done adding a number to a signal. The addition shifts the value of every sample up (or down) by the same amount.

        Parameters
        ----------
        offsetted_shape : (int,int)
            To set the offset value of the wave.

        ceil : int
            maximum read value

        Returns
        -------
        int
            value read from the readout task
        """
        return self.readout_task.read(offsetted_shape, ceil)

    def remote_read(self, offsetted_shape, ceil):
        """
        Read from the readout channel but with an offset value which can be specified. Th function also takes a ceil value as a parameter, therefore, all reads are under this maximum value.
        Offset on the read is done adding a number to a signal. The addition shifts the value of every sample up (or down) by the same amount.
        This method helps you read tasks from a machine remotely and throws an error if it is unable to read from certain task ( and will return -1 - Exit )

        Parameters
        ----------
        offsetted_shape : (int,int)
            To set the offset value of the wave.
        ceil : int
            maximum read value

        Returns
        -------
        int
            value read from the readout task
        """
        try:
            return self.readout_task.read(offsetted_shape, ceil)
        except nidaqmx.errors.DaqError as e:
            print("Error reading: " + str(e))
        return -1

    @Pyro4.oneway
    def start_trigger(self, trigger_source):
        """
        To start triggering to the device.
        The name of the trigger source for this device should be specified.
        The trigger source setting of the instrument determines which trigger signals are used to trigger the instrument.
        The trigger source can be set to a single channel or to any combination of channels or other trigger sources.

        Parameters
        ----------
        trigger_source: str
            source trigger name
        """
        self.activation_task.triggers.start_trigger.cfg_dig_edge_start_trig(
            "/" + trigger_source + "/ai/StartTrigger"
        )

    @Pyro4.oneway
    def remote_start_tasks(self, y, auto_start):
        """
        To start the tasks on this device remotely .The method invokes 2 other methods:

        1. self.activation_task.start() - To start the activation tasks on the device
        2. self.readout_task.start() - To start the readout tasks on the device.

        Both of these tasks occur simulataneously and are synchronized.
        The tasks will start automatically if the "auto_start" option is set to True in the configuration dictionary used to intialize this device.

        Parameters
        ----------
        y : list
            list of tasks
        auto_start : Bool
            True or False based on wheather to auto start the tasks or not
        """
        self.activation_task.write(np.asarray(y), auto_start=auto_start)
        if not auto_start:
            self.activation_task.start()
            self.readout_task.start()

    @Pyro4.oneway
    def start_tasks(self, y, auto_start):
        """
        To start the tasks on this device locally.The method invokes 2 other methods:

        1. self.activation_task.start() - To start the activation tasks on the device
        2. self.readout_task.start() - To start the readout tasks on the device.

        Both of these tasks occur simulataneously and are synchronized.
        The tasks will start automatically if the "auto_start" option is set to True in the configuration dictionary used to intialize this device.

        Parameters
        ----------
        y : list
            list of tasks
        auto_start : Bool
            True or False based on wheather to auto start the tasks or not
        """
        y = np.require(y, dtype=y.dtype, requirements=["C", "W"])
        try:
            self.activation_task.write(y, auto_start=auto_start)
        except nidaqmx.errors.DaqError:
            print(
                "There was an error writing to the activation task: "
                + self.activation_task.name
            )
            print("Trying to reset device and do the read again.")
            for dev in self.devices:
                dev.reset_device()
            self.init_activation_channels(
                self.activation_channel_names, self.voltage_ranges
            )
            self.activation_task.write(y, auto_start=auto_start)

        if not auto_start:
            self.activation_task.start()
            self.readout_task.start()

    @Pyro4.oneway
    def stop_tasks(self):
        """
        To stop the all tasks on this device namely - the activation tasks and the readout tasks to and from the device.
        The tasks can be started again if required by using the start_tasks() method.
        """
        self.readout_task.stop()
        self.activation_task.stop()

    def init_tasks(self, configs):
        """
        To Initialize the tasks on the device based on the configurations dictionary provided.
        the method initializes the activation and readout tasks from the device by setting the voltage ranges and choosing the instruments that have been specified.

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
        self.init_activation_channels(
            self.activation_channel_names, self.voltage_ranges
        )
        self.init_readout_channels(self.readout_channel_names)
        self.set_sampling_clocks(self.configs["sampling_frequency"], 
                                 self.configs['instruments_setup']['trigger_source'])
        return self.voltage_ranges.tolist()

    @Pyro4.oneway
    def close_tasks(self):
        """
        Close all the task on this device -  both activation and readout tasks by deleting them.
        Note - This method is different from the stop_tasks() method which only stops the current tasks temporarily.
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
    This class acts as client program that calls methods on the Pyro object - LocalTasks.
    The methods from this class can be used to access these tasks remotely.
    The remote method calls on Pyro objects from LocalTasks go through a proxy.Therefore the RemoteServer should be initialized.
    The proxy can be treated as if it was the actual object, so you write normal python code to call the remote methods and deal with the return values, or even exceptions:
    """

    def __init__(self, uri):
        """
        sets up the proxy server so that remote method calls can be made to the device. This enables calling methods on the Pyro object - LocalTasks.
        It also initializes the device to Acquire or generate a finite number of samples.

        Parameters
        ----------
        uri : str
            uri of the remote server
        """
        self.acquisition_type = constants.AcquisitionType.FINITE
        self.tasks = Pyro4.Proxy(uri)

    def init_activation_channels(self, channel_names, voltage_ranges=None):
        """
        Initialises the activation electrodes of the device.
        The output of the computer is the input of the device.
        These are being sent as list of voltage values and channel names which are the start values for device. They can however be tuned later according to requirements.
        Activation electrodes -
            Range - 1.2 to 0.6V or -0.7 to 0.3V​
            They have P-n junction forward bias​.Forward bias occurs when a voltage is applied such that the electric field formed by the P-N junction is decreased.
            If it is outside the range, there are Noisy solutions which are defined in the noise.py class.

        Parameters
        ----------
        channel_names : list
            list of the names of the activation channels for this device
        voltage_ranges : list, optional
            list of voltage ranges for this device, by default None
        """
        self.tasks.init_activation_channels(channel_names, voltage_ranges)

    def init_readout_channels(self, readout_channels):
        """
        Initializes the readout instrument of the device.
        The input of the computer which is the output of the device
        The range of the readout channels depends on setup​ and on the feedback resistance produced.It also has clipping ranges​ which can be set according to preference.
        Example ranges -400 to 400 or -100 to 100 nA​.

        Parameters
        ----------
        readout_channels : list
            list of readout channes for the device
        """
        self.tasks.init_readout_channels(readout_channels)

#    def set_shape(self, sampling_frequency, shape):
#        """
#        One way method to set the shape variables for the data that is being sent to the device.
#        Depending on which device is being used, CDAQ or NIDAQ, and the sampling frequency, the shape of the data that is being sent can to be specified.
#       This function helps to tackle the problem of differnt batches having differnt data shapes (for example - differnt sample size) when dealing with big data.
#
#        Parameters
#        ----------
#        sampling_frequency : float
#             the average number of samples to be obtained in one second
#        shape : (int,int)
#            required shape of for sampling
#        """
#        self.tasks.set_shape(sampling_frequency, shape)

    def add_synchronisation_channels(
        self,
        readout_instrument,
        activation_instrument,
        activation_channel_no=7,
        readout_channel_no=7,
    ):
        """
        The method is used to add a synchronized activation and readout channel to the device.
        Channels can be used to synchronize goroutines which are read/write tasks on the device.
        A channel can make a goroutine wait until its finished. Sometimes a goroutine needs to be finished before you can start the next one (synchronous). This can be solved with channels.

        Parameters
        ----------
        readout_instrument: str
            name of the readout instrument
        activation_instrument : str
            name of the activation instrument
        activation_channel_no : int, optional
            Channel through which voltages will be sent for activating the device (with both data inputs and control voltages), by default 7
        readout_channel_no : int, optional
            Channel for reading the output current values, by default 7
        """
        self.tasks.add_synchronisation_channels(
            readout_instrument,
            activation_instrument,
            activation_channel_no,
            readout_channel_no,
        )

    def read(self, offsetted_shape, ceil):
        """
        Read from the readout channel but with an offset value which can be specified. Th function alos takes a ceil value as a parameter, therefore, all reads are under this maximum value.
        Offset on the read is done adding a number to a signal. The addition shifts the value of every sample up (or down) by the same amount.

        Parameters
        ----------
        offsetted_shape : (int,int)
            shape based on the offset value
        ceil : int
            max read value

        Returns
        -------
        int
            task read from the readout channel
        """
        return self.tasks.remote_read(offsetted_shape, ceil)

    def start_trigger(self, trigger_source):
        """
        To start triggering to the device.
         The name of the trigger source for this device should be specified.
         The trigger source setting of the instrument determines which trigger signals are used to trigger the instrument.
         The trigger source can be set to a single channel or to any combination of channels or other trigger sources.

         Parameters
         ----------
         trigger_source: str
             source trigger name
        """
        self.tasks.start_trigger(trigger_source)

    def start_tasks(self, y, auto_start):
        """
        To start the tasks on this device remotely .The method invokes 2 other methods:

        1. self.activation_task.start() - To start the activation tasks on the device
        2. self.readout_task.start() - To start the readout tasks on the device.

        Both of these tasks occur simulataneously and are synchronized.
        The tasks will start automatically if the "auto_start" option is set to True in the configuration dictionary used to intialize this device.


        Parameters
        ----------
        y : list
            list of tasks
        auto_start : Bool
            True or False based on wheather to auto start the tasks or not
        """
        self.tasks.remote_start_tasks(y.tolist(), auto_start)

    def stop_tasks(self):
        """
        To stop the all tasks on this device namely - the activation tasks and the readout tasks to and from the device.
        The tasks can be started again if required by using the start_tasks() method
        """
        self.tasks.stop_tasks()

    def init_tasks(self, configs):
        """
        To Initialize the tasks on the device based on the configurations dictionary provided.
        the method initializes the activation and readout tasks from the device by setting the voltage ranges and choosing the instruments that have been specified.

        Parameters
        ----------
        configs : dict
            configs dictionary for the device model

        Returns
        -------
        list
            list of voltage ranges
        """
        self.voltage_ranges = np.array(self.tasks.init_tasks(configs))

    def close_tasks(self):
        """
        Close all tasks
        """
        self.tasks.close_tasks()


class RemoteTasksServer:
    """
    Server for the Remote Tasks on the device.
    To be able to call methods on a Pyro object , an appropriate URI is created and given to the server.
    The class uses the Pyro remote server. It enables objects between LocalTasks and RemoteTasks to talk to each other over the network.
    This class provides the object -  (LocalTasks object) and actually runs the methods.

    """

    def __init__(self, configs):
        """
        Initialize the remote server based on the configurations of the model and initialize the LocalTasks.
        Remote method calls can be made after the server is initialized.

        Parameters
        ----------
        configs : dict
            configs of the model
        """
        self.configs = configs
        self.tasks = LocalTasks()

    def save_uri(self, uri):
        """
        To give this remote server a URI which will be saved in a text file.


        Parameters
        ----------
        uri : str
            uri of the server
        """
        print("Server ready, object URI: " + str(uri))
        f = open("uri.txt", "w")
        f.write(str(uri) + " \n")
        f.close()

    def start(self):
        """
        Starts a daemon thread to handle to the remote server and the remote tasks on the device.
        A thread is a separate flow of execution. It enables getting multiple tasks running simultaneously.
        Therefore, read and write tasks of the device can happen concurrently.
        This method starts a daemon thread which will shut down immediately when the program exits.

        """
        self.daemon = Pyro4.Daemon(host=self.configs["ip"], port=self.configs["port"])
        uri = self.daemon.register(self.tasks)
        self.save_uri(uri)
        self.daemon.requestLoop()

    def stop(self):
        """
        Method to stop the remote server by closing the daemon thread.
        """
        self.daemon.close()


def deploy_driver(configs):
    """
    Deploy the driver based on the configs and start the remote server.
    The method initializes the IP address,port number,subnet mask to the Default values at the top of the file.
    To access remote objects - LocalTasks on the device - (Cdaq or Nidaq) , the method starts the RemoteTasks server on these default configurations.

    The remote server runs on this default IP address. An IP address is used in order to uniquely identify a device on an IP network.
    The address is divided into a network portion and host portion with the help of a subnet mask.
    A port number is always associated with an IP address.There ar multiple ports and we use the 8081 port (can be changed)

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


if __name__ == "__main__":
    deploy_driver({})
