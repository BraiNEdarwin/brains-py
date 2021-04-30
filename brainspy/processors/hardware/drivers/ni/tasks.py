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
    get the tasks driver based on the configurations dictionary

    Parameters
    ----------
    configs : dict
        configurations of the device model

    Returns
    -------
    class - RemoteTasks() or LocalTasks() based on configs
        [description]
    """
    if configs["real_time_rack"]:
        return RemoteTasks(configs["uri"])
    else:
        return LocalTasks()


def run_server(configs):
    """
    To start and run the server for the RemoteTasks

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
    Give a static IP address to the system with a mask

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
    Classes and methods available for remote access.
    """

    def __init__(self):
        """
        Initialize the activation and readout task as Null values in the beginning
        """
        self.acquisition_type = constants.AcquisitionType.FINITE
        self.activation_task = None
        self.readout_task = None

    @Pyro4.oneway
    def init_activation_channels(self, channel_names, voltage_ranges=None):
        """
        Initialises the output of the computer which is the input of the device

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
        Initialises the input of the computer which is the output of the device

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
    def set_shape(self, sampling_frequency, shape):
        """
        One way method to set the shape vars for the device based on the sampling frequency as defined in the configs dictionary

        Parameters
        ----------
        sampling_frequency : float
             the average number of samples to be obtained in one second
        shape : (int,int)
            required shape of for sampling
        """
        self.activation_task.timing.cfg_samp_clk_timing(
            sampling_frequency,
            sample_mode=self.acquisition_type,
            samps_per_chan=shape,
        )
        self.readout_task.timing.cfg_samp_clk_timing(
            sampling_frequency,
            sample_mode=self.acquisition_type,
            samps_per_chan=shape,  # TODO: Add shape + 1 ?
        )

    @Pyro4.oneway
    def add_synchronisation_channels(
        self,
        readout_instrument,
        activation_instrument,
        activation_channel_no=7,
        readout_channel_no=7,
    ):
        """
        adds a synchronization channel by specifying an activation task and doing a synchronized readout

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
        read from the readout channel

        Parameters
        ----------
        offsetted_shape : (int,int)
            shape based on the offset value
        ceil : int
            max read value

        Returns
        -------
        Task
            task read from the readout channel
        """
        return self.readout_task.read(offsetted_shape, ceil)

    def remote_read(self, offsetted_shape, ceil):
        """
        Remotely read the task from the readout channel

        Parameters
        ----------
        offsetted_shape : (int,int)
            shape based on the offset value
        ceil : int
            max read value

        Returns
        -------
        Task
            task read from the readout channel
        """
        try:
            return self.readout_task.read(offsetted_shape, ceil)
        except nidaqmx.errors.DaqError as e:
            print("Error reading: " + str(e))
        return -1

    @Pyro4.oneway
    def start_trigger(self, trigger_source):
        """
        One way task to start the trigger the task

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
        Remotely start the tasks by starting the activation task and readout task which are synchronized
        This is a one way task.

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
        """start the tasks by starting the activation task and readout task which are synchronized
        This is a one way task.

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
        stop all activation and readout tasks
        """
        self.readout_task.stop()
        self.activation_task.stop()

    def init_tasks(self, configs):
        """
        Initialize the tasks based on the configurations dictionary

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
        return self.voltage_ranges.tolist()

    @Pyro4.oneway
    def close_tasks(self):
        """
        Close all tasks
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
    Class to initialize and handle the tasks on the device from the remote server.
    """

    def __init__(self, uri):
        """
        Initialize the tasks from the remote server

        Parameters
        ----------
        uri : str
            uri of the remote server
        """
        self.acquisition_type = constants.AcquisitionType.FINITE
        self.tasks = Pyro4.Proxy(uri)

    def init_activation_channels(self, channel_names, voltage_ranges=None):
        """
        Initialises the output of the computer which is the input of the device

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
        Initialises the input of the computer which is the output of the device

        Parameters
        ----------
        readout_channels : list
            list of readout channes for the device
        """
        self.tasks.init_readout_channels(readout_channels)

    def set_shape(self, sampling_frequency, shape):
        """
        One way method to set the shape vars for the device based on the sampling frequency as defined in the configs dictionary

        Parameters
        ----------
        sampling_frequency : float
             the average number of samples to be obtained in one second
        shape : (int,int)
            required shape of for sampling
        """
        self.tasks.set_shape(sampling_frequency, shape)

    def add_synchronisation_channels(
        self,
        readout_instrument,
        activation_instrument,
        activation_channel_no=7,
        readout_channel_no=7,
    ):
        """
        adds a synchronization channel by specifying an activation task and doing a synchronized readout

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
        read from the readout channel

        Parameters
        ----------
        offsetted_shape : (int,int)
            shape based on the offset value
        ceil : int
            max read value

        Returns
        -------
        Task
            task read from the readout channel
        """
        return self.tasks.remote_read(offsetted_shape, ceil)

    def start_trigger(self, trigger_source):
        """
        One way task to start the trigger the task

        Parameters
        ----------
        trigger_source: str
            source trigger name
        """
        self.tasks.start_trigger(trigger_source)

    def start_tasks(self, y, auto_start):
        """
        start the tasks by starting the activation task and readout task which are synchronized
        This is a one way task.

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
        stop all activation and readout tasks
        """
        self.tasks.stop_tasks()

    def init_tasks(self, configs):
        """
        Initialize the tasks based on the configurations dictionary

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
    The class uses the Pyro remote server. It is a library that enables objects to talk to each other over the network
    """

    def __init__(self, configs):
        """
        Initialize the remote server based on the configurations o the model

        Parameters
        ----------
        configs : dict
            configs of the model
        """
        self.configs = configs
        self.tasks = LocalTasks()

    def save_uri(self, uri):
        """
        Save the uri of this remote server

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
        Starts a daemon thread to handle to the remote server and the remote tasks on the device
        """
        self.daemon = Pyro4.Daemon(host=self.configs["ip"], port=self.configs["port"])
        uri = self.daemon.register(self.tasks)
        self.save_uri(uri)
        self.daemon.requestLoop()

    def stop(self):
        self.daemon.close()


def deploy_driver(configs):
    """Deploy the driver based on the configs and start the server

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
