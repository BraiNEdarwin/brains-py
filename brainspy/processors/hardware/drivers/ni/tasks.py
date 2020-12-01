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
    if configs["tasks_driver_type"] == "local":
        return LocalTasks()
    elif configs["tasks_driver_type"] == "remote":
        return RemoteTasks(configs["uri"])
    else:
        raise NotImplementedError(
            f"{configs['tasks_driver_type']} 'tasks_driver_type' configuration is not recognised. The driver type has to be defined as 'local' or 'remote'. "
        )


def run_server(configs):
    if configs["force_static_ip"]:
        set_static_ip(configs["server"])
    server = RemoteTasksServer(configs)
    server.start()


def set_static_ip(configs):
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
    def __init__(self):
        self.acquisition_type = constants.AcquisitionType.FINITE
        self.activation_task = None
        self.readout_task = None

    @Pyro4.oneway
    def init_activation_channels(
        self, channel_names, voltage_ranges=None
    ):
        """Initialises the output of the computer which is the input of the device"""
        self.activation_task = nidaqmx.Task('activation_task_' + datetime.utcnow().strftime('%Y_%m_%d_%H%M%S_%f'))
        for i in range(len(channel_names)):
            channel_name = str(channel_names[i])
            if voltage_ranges is not None:
                assert voltage_ranges[i][0] > -2 and voltage_ranges[i][0] < 2, "Minimum voltage ranges configuration outside of the allowed values -2 and 2"
                assert voltage_ranges[i][1] > -2 and voltage_ranges[i][1] < 2, "Maximum voltage ranges configuration outside of the allowed values -2 and 2"
                self.activation_task.ao_channels.add_ao_voltage_chan(
                    #activation_instrument + "/ai" + str(activation_channels[i])
                    channel_name, min_val=voltage_ranges[i][0].item() - RANGE_MARGIN, max_val=voltage_ranges[i][1].item() + RANGE_MARGIN
                )
            else:
                print('WARNING! READ CAREFULLY THIS MESSAGE. Activation channels have been initialised without a security voltage range, they will be automatically set up to a range between -2 and 2 V. This may result in damaging the device. Press ENTER only if you are sure that you want to proceed, otherwise STOP the program. Voltage ranges can be defined in the instruments setup configurations.')
                input()
                self.activation_task.ao_channels.add_ao_voltage_chan(
                    #activation_instrument + "/ai" + str(activation_channels[i])
                    channel_name, min_val=-2, max_val=2
                )

    @Pyro4.oneway
    def init_readout_channels(
        self, readout_channels
    ):
        """Initialises the input of the computer which is the output of the device"""
        self.readout_task = nidaqmx.Task('readout_task_' + datetime.utcnow().strftime('%Y_%m_%d_%H%M%S_%f'))
        for i in range(len(readout_channels)):
            channel = readout_channels[i]
            self.readout_task.ai_channels.add_ai_voltage_chan(
                # activation_instrument + "/ao" + str(activation_channels[i]),
                # "ao" + str(i) + "",
                # -2,
                # 2,
                channel
            )

    @Pyro4.oneway
    def set_shape(self, sampling_frequency, shape):
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
    def add_synchronisation_channels(self, readout_instrument, activation_instrument, activation_channel_no=7, readout_channel_no=7):
        # Define ao7 as sync signal for the NI 6216 ai0
        self.activation_task.ao_channels.add_ao_voltage_chan(
            activation_instrument + "/ao" + str(activation_channel_no), name_to_assign_to_channel="activation_synchronisation_channel", min_val=-5, max_val=5
        )
        self.readout_task.ai_channels.add_ai_voltage_chan(
            readout_instrument + "/ai" + str(readout_channel_no), name_to_assign_to_channel="readout_synchronisation_channel", min_val=-5, max_val=5
        )

    def read(self, offsetted_shape, ceil):
        return self.readout_task.read(offsetted_shape, ceil)

    def remote_read(self, offsetted_shape, ceil):
        try:
            return self.readout_task.read(offsetted_shape, ceil)
        except nidaqmx.errors.DaqError as e:
            print("Error reading: " + str(e))
        return -1

    @Pyro4.oneway
    def start_trigger(self, trigger_source):
        self.activation_task.triggers.start_trigger.cfg_dig_edge_start_trig(
            "/" + trigger_source + "/ai/StartTrigger"
        )

    @Pyro4.oneway
    def remote_start_tasks(self, y, auto_start):
        self.activation_task.write(np.asarray(y), auto_start=auto_start)
        if not auto_start:
            self.activation_task.start()
            self.readout_task.start()

    @Pyro4.oneway
    def start_tasks(self, y, auto_start):
        y = np.require(y, dtype=y.dtype, requirements=["C", "W"])
        try:
            self.activation_task.write(y, auto_start=auto_start)
        except nidaqmx.errors.DaqError as e:
            print('There was an error writing to the activation task: ' + self.activation_task.name)
            print('Trying to reset device and do the read again.')
            for dev in self.devices:
                dev.reset_device()
            self.init_activation_channels(self.activation_channel_names, self.voltage_ranges)
            self.activation_task.write(y, auto_start=auto_start)

        if not auto_start:
            self.activation_task.start()
            self.readout_task.start()

    @Pyro4.oneway
    def stop_tasks(self):
        self.readout_task.stop()
        self.activation_task.stop()

    @Pyro4.oneway
    def init_tasks(self, configs):
        self.configs = configs
        self.activation_channel_names, self.readout_channel_names, instruments, self.voltage_ranges = init_channel_data(configs)
        devices = []
        for instrument in instruments:
            devices.append(device.Device(name=instrument))
        self.devices = devices
        # TODO: add a maximum and a minimum to the activation channels
        self.init_activation_channels(self.activation_channel_names, self.voltage_ranges)
        self.init_readout_channels(self.readout_channel_names)
        return self.voltage_ranges

    @Pyro4.oneway
    def close_tasks(self):
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
    def __init__(self, uri):
        self.acquisition_type = constants.AcquisitionType.FINITE
        self.tasks = Pyro4.Proxy(uri)
        self.close_tasks()

    def init_activation_channels(self, channel_names, voltage_ranges=None):
        self.tasks.init_activation_channels(channel_names, voltage_ranges)

    def init_readout_channels(self, readout_channels):
        self.tasks.init_readout_channels(readout_channels)

    def set_shape(self, sampling_frequency, shape):
        self.tasks.set_shape(sampling_frequency, shape)

    def add_synchronisation_channels(self, readout_instrument, activation_instrument, activation_channel_no=7, readout_channel_no=7):
        self.tasks.add_synchronisation_channels(readout_instrument, activation_instrument, activation_channel_no, readout_channel_no)

    def read(self, offsetted_shape, ceil):
        return self.tasks.remote_read(offsetted_shape, ceil)

    def start_trigger(self, trigger_source):
        self.tasks.start_trigger(trigger_source)

    def start_tasks(self, y, auto_start):
        self.tasks.remote_start_tasks(y.tolist(), auto_start)

    def stop_tasks(self):
        self.tasks.stop_tasks()

    def init_tasks(self, configs):
        self.voltage_ranges = self.tasks.init_tasks(configs)

    def close_tasks(self):
        self.tasks.close_tasks()


class RemoteTasksServer:
    def __init__(self, configs):
        self.configs = configs
        self.tasks = LocalTasks()

    def save_uri(self, uri):
        print("Server ready, object URI: " + str(uri))
        f = open("uri.txt", "w")
        f.write(str(uri) + " \n")
        f.close()

    def start(self):
        self.daemon = Pyro4.Daemon(host=self.configs["ip"], port=self.configs["port"])
        uri = self.daemon.register(self.tasks)
        self.save_uri(uri)
        self.daemon.requestLoop()

    def stop(self):
        self.daemon.close()


def deploy_driver(configs):
    configs["ip"] = DEFAULT_IP
    configs["port"] = DEFAULT_PORT
    configs["subnet_mask"] = DEFAULT_SUBNET_MASK
    configs["force_static_ip"] = False

    run_server(configs)


if __name__ == "__main__":
    deploy_driver({})
