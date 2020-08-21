import os
import time
import numpy as np

import nidaqmx
import nidaqmx.constants as constants
import Pyro4

DEFAULT_IP = '192.168.1.5'
DEFAULT_SUBNET_MASK = '255.255.255.0'
DEFAULT_PORT = 8081

SWITCH_ETHERNET_OFF_COMMAND = "ifconfig eth0 down"
SWITCH_ETHERNET_ON_COMMAND = "ifconfig eth0 up"


def get_driver(configs):
    if configs['driver_type'] == 'local':
        return LocalTasks()
    elif configs['driver_type'] == 'remote':
        return RemoteTasks(configs['uri'])
    else:
        raise NotImplementedError(f"{configs['driver_type']} 'driver_type' configuration is not recognised. The driver type has to be defined as 'local' or 'remote'. ")


def run_server(configs):
    if configs['force_static_ip']:
        set_static_ip(configs['server'])
    server = RemoteTasksServer(configs)
    server.start()


def set_static_ip(configs):
    SET_STATIC_IP_COMMAND = f"ifconfig eth0 {configs['ip']} netmask {configs['subnet_mask']} up"
    os.system(SET_STATIC_IP_COMMAND)
    time.sleep(1)
    os.system(SWITCH_ETHERNET_OFF_COMMAND)
    time.sleep(1)
    os.system(SWITCH_ETHERNET_ON_COMMAND)


@Pyro4.expose
class LocalTasks():
    def __init__(self):
        self.acquisition_type = constants.AcquisitionType.FINITE
        self.output_task = None
        self.input_task = None

    @Pyro4.oneway
    def init_output(self, input_channels, output_instrument, sampling_frequency, offsetted_shape):
        '''Initialises the output of the computer which is the input of the device'''
        self.output_task = nidaqmx.Task()
        for i in range(len(input_channels)):
            self.output_task.ao_channels.add_ao_voltage_chan(output_instrument + '/ao' + str(input_channels[i]), 'ao' + str(i) + '', -2, 2)
        self.output_task.timing.cfg_samp_clk_timing(sampling_frequency, sample_mode=self.acquisition_type, samps_per_chan=offsetted_shape)

    @Pyro4.oneway
    def init_input(self, output_channels, input_instrument, sampling_frequency, offsetted_shape):
        '''Initialises the input of the computer which is the output of the device'''
        self.input_task = nidaqmx.Task()
        for i in range(len(output_channels)):
            self.input_task.ai_channels.add_ai_voltage_chan(input_instrument + '/ai' + str(output_channels[i]))
        self.input_task.timing.cfg_samp_clk_timing(sampling_frequency, sample_mode=self.acquisition_type, samps_per_chan=offsetted_shape)

    @Pyro4.oneway
    def add_channels(self, output_instrument, input_instrument):
        # Define ao7 as sync signal for the NI 6216 ai0
        self.output_task.ao_channels.add_ao_voltage_chan(output_instrument + '/ao7', 'ao7', -5, 5)
        self.input_task.ai_channels.add_ai_voltage_chan(input_instrument + '/ai7')

    def read(self, offsetted_shape, ceil):
        return self.input_task.read(offsetted_shape, ceil)

    def remote_read(self, offsetted_shape, ceil):
        try:
            return self.input_task.read(offsetted_shape, ceil)
        except nidaqmx.errors.DaqError as e:
            print('Error reading: ' + str(e))
        return -1

    @Pyro4.oneway
    def start_trigger(self, trigger_source):
        self.output_task.triggers.start_trigger.cfg_dig_edge_start_trig('/' + trigger_source + '/ai/StartTrigger')

    @Pyro4.oneway
    def remote_start_tasks(self, y, auto_start):
        self.output_task.write(np.asarray(y), auto_start=auto_start)
        if not auto_start:
            self.output_task.start()
            self.input_task.start()

    @Pyro4.oneway
    def start_tasks(self, y, auto_start):
        y = np.require(y, dtype=y.dtype, requirements=['C', 'W'])
        self.output_task.write(y, auto_start=auto_start)
        if not auto_start:
            self.output_task.start()
            self.input_task.start()

    @Pyro4.oneway
    def stop_tasks(self):
        self.input_task.stop()
        self.output_task.stop()

    @Pyro4.oneway
    def close_tasks(self):
        if self.input_task is not None:
            self.input_task.close()
        if self.output_task is not None:
            self.output_task.close()


class RemoteTasks():
    def __init__(self, uri):
        self.acquisition_type = constants.AcquisitionType.FINITE
        self.tasks = Pyro4.Proxy(uri)
        self.close_tasks()

    def init_output(self, input_channels, output_instrument, sampling_frequency, offsetted_shape):
        self.tasks.init_output(input_channels, output_instrument, sampling_frequency, offsetted_shape)

    def init_input(self, output_channels, input_instrument, sampling_frequency, offsetted_shape):
        self.tasks.init_input(output_channels, input_instrument, sampling_frequency, offsetted_shape)

    def add_channels(self, output_instrument, input_instrument):
        self.tasks.add_channels(output_instrument, input_instrument)

    def read(self, offsetted_shape, ceil):
        return self.tasks.remote_read(offsetted_shape, ceil)

    def start_trigger(self, trigger_source):
        self.tasks.start_trigger(trigger_source)

    def start_tasks(self, y, auto_start):
        self.tasks.remote_start_tasks(y.tolist(), auto_start)

    def stop_tasks(self):
        self.tasks.stop_tasks()

    def close_tasks(self):
        self.tasks.close_tasks()


class RemoteTasksServer():
    def __init__(self, configs):
        self.configs = configs
        self.tasks = LocalTasks()

    def save_uri(self, uri):
        print('Server ready, object URI: ' + str(uri))
        f = open('uri.txt', 'w')
        f.write(str(uri) + ' \n')
        f.close()

    def start(self):
        self.daemon = Pyro4.Daemon(host=self.configs['ip'], port=self.configs['port'])
        uri = self.daemon.register(self.tasks)
        self.save_uri(uri)
        self.daemon.requestLoop()

    def stop(self):
        self.daemon.close()


def deploy_driver(configs):
    configs['ip'] = DEFAULT_IP
    configs['port'] = DEFAULT_PORT
    configs['subnet_mask'] = DEFAULT_SUBNET_MASK
    configs['force_static_ip'] = False

    run_server(configs)


if __name__ == "__main__":
    deploy_driver({})
