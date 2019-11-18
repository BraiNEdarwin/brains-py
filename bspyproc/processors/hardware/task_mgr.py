import os

DEFAULT_IP = '192.168.1.5'
DEFAULT_SUBNET_MASK = '255.255.255.0'
DEFAULT_PORT = 1080

SWITCH_ETHERNET_OFF_COMMAND = "ifconfig eth0 down"
SWITCH_ETHERNET_ON_COMMAND = "ifconfig eth0 up"
def get_driver(configs):
    if configs['driver']['driver_type'] == 'local':
        return LocalTasks(configs)
    elif configs['driver']['driver_type'] == 'remote':
        return RemoteTasksClient(configs)
    else:
       raise NotImplementedError(f"{configs['driver_type']} 'driver_type' configuration is not recognised. The driver type has to be defined as 'local' or 'remote'. ")

def run_server(ip=DEFAULT_IP, subnet_mask=DEFAULT_SUBNET_MASK, port=DEFAULT_PORT, set_static_ip=True):
    if set_static_ip:
        set_static_ip(ip,subnet_mask)
    server = RemoteTasksServer(ip, port)
    server.start()

def set_static_ip(ip=DEFAULT_IP, subnet_mask=DEFAULT_SUBNET_MASK):
    SET_STATIC_IP_COMMAND = f"ifconfig eth0 {ip} netmask {subnet_mask} up"
    os.system(SET_STATIC_IP_COMMAND)
    os.wait(1)
    os.system(SWITCH_ETHERNET_OFF_COMMAND)
    os.wait(1)
    os.system(SWITCH_ETHERNET_ON_COMMAND)


class LocalTasks():
    import nidaqmx
    import nidaqmx.constants as constants
    def __init__(self, configs):
        self.configs = configs
        self.acquisition_type = constants.AcquisitionType.FINITE

    def get_task(self):
        return nidaqmx.Task()
    
    def init_output(self):
        '''Initialises the output of the computer which is the input of the device'''

        self.output_task = self.driver.get_task()
        for i in range(len(self.configs['input_channels'])):
            self.output_task.ao_channels.add_ao_voltage_chan(self.configs['output_instrument'] + '/ao' + str(self.configs['input_channels'][i]), 'ao' + str(i) + '', -2, 2)
        self.output_task.timing.cfg_samp_clk_timing(self.configs['sampling_frequency'], sample_mode=self.acquisition_type, samps_per_chan=self.configs['shape'] + self.configs['offset'])

    def init_input(self):
        '''Initialises the input of the computer which is the output of the device'''

        self.input_task = self.driver.get_task()
        for i in range(len(self.configs['output_channels'])):
            self.input_task.ai_channels.add_ai_voltage_chan(self.configs['input_instrument'] + '/ai' + str(self.configs['output_channels'][i]))
        self.input_task.timing.cfg_samp_clk_timing(self.configs['sampling_frequency'], sample_mode=self.acquisition_type, samps_per_chan=self.configs['shape'] + self.configs['offset'])

    def start_tasks(self, y):
        self.output_task.write(y, auto_start=self.configs['auto_start'])
        if not self.configs['auto_start']:
            self.output_task.start()
            self.input_task.start()

    def stop_tasks(self):
        self.input_task.stop()
        self.output_task.stop()

    def close_tasks(self):
        self.input_task.close()
        self.output_task.close()

class RemoteTasksClient():
    import Pyro4
    def __init__(self, uri):
        self.tasks = Pyro4.Proxy(uri) 

    def get_task(self):
        return self.tasks.get_task()
    
    def init_output(self):
        self.tasks.init_output()

    def init_input(self):
        self.tasks.init_input()

    def start_tasks(self, y):
        self.tasks.start_tasks(y)

    def stop_tasks(self):
        self.tasks.stop_tasks()

    def close_tasks(self):
        self.tasks.close_tasks()

    def shutdown(self):
        self.tasks.close_tasks()
        self.tasks.shutdown()

class RemoteTasksServer():
    import Pyro4
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.tasks = LocalTasks()

    def save_uri(self, uri):
        print('Server ready, object URI: '+uri)
        f = open('uri.txt')
        f.write(str(uri)+' \n')
        f.close()

    def start(self):
        self.daemon = Pyro4.Daemon(host=self.ip, port=self.port)
        uri = self.daemon.register(self.tasks)
        self.save_uri(uri)
        self.daemon.requestLoop()

    def stop(self):
        self.daemon.close()

if __name__ == "__main__":
    run_server()


    

    
    
    

    

