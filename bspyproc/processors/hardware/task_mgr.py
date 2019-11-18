import os
import time

import nidaqmx
import nidaqmx.constants as constants

import Pyro4

DEFAULT_IP = '192.168.1.5'
DEFAULT_SUBNET_MASK = '255.255.255.0'
DEFAULT_PORT = 8081

SWITCH_ETHERNET_OFF_COMMAND = "ifconfig eth0 down"
SWITCH_ETHERNET_ON_COMMAND = "ifconfig eth0 up"

def get_driver(configs):
    if configs['driver']['driver_type'] == 'local':
        return LocalTasks()
    elif configs['driver']['driver_type'] == 'remote':
        return RemoteTasksClient(configs)
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


class LocalTasks():
    def __init__(self):
        self.acquisition_type = constants.AcquisitionType.FINITE

    def get_task(self):
        return nidaqmx.Task()
    

class RemoteTasksClient():
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
    def __init__(self, configs):
        self.configs = configs
        self.tasks = LocalTasks()

    def save_uri(self, uri):
        print('Server ready, object URI: '+str(uri))
        f = open('uri.txt', 'w')
        f.write(str(uri)+' \n')
        f.close()

    def start(self):
        self.daemon = Pyro4.Daemon(host=self.configs['ip'], port=self.configs['port'])
        uri = self.daemon.register(self.tasks)
        self.save_uri(uri)
        self.daemon.requestLoop()

    def stop(self):
        self.daemon.close()

if __name__ == "__main__":
    configs = {}
    configs['ip'] = DEFAULT_IP
    configs['port'] = DEFAULT_PORT
    configs['subnet_mask'] = DEFAULT_SUBNET_MASK
    configs['force_static_ip'] = False

    run_server(configs)


    

    
    
    

    

