'''

'''
import numpy as np
import math

from  bspyproc.processors.hardware import task_mgr

class NationalInstrumentsSetup():

    def __init__(self, reset_mgr, configs):
        self.configs = configs
        self.driver = task_mgr.get_driver(configs)
        self.offsetted_shape = configs['shape'] + configs['offset']
        self.ceil = math.ceil((self.offsetted_shape) / self.configs['sampling_frequency']) + 1
        self.driver.init_output()
        self.driver.init_input()

    def process_output_data(self, data):
        data = np.asarray(data)
        if len(data.shape) == 1:
            data = data[np.newaxis, :]
        return data


    def read_data(self, y):
        '''
            y = It represents the input data as matrix where the shpe is defined by the "number of inputs to the device" times "input points that you want to input to the device".
        '''
        self.driver.start_tasks(y)
        read_data = self.driver.input_task.read(self.offsetted_shape, self.ceil)
        self.driver.stop_tasks()

        return read_data

    def close():
        self.driver.close_tasks()


class CDAQtoCDAQ(NationalInstrumentsSetup):

    def __init__(self, configs):
        configs['auto_start'] = True
        configs['offset'] = 0
        super().__init__(self, configs)
        self.driver.output_task.triggers.start_trigger.cfg_dig_edge_start_trig('/' + self.configs['trigger_source'] + '/ai/StartTrigger')

    def get_output(self, y):
        y = y.T
        assert self.configs['shape'] == y.shape[1]
        data = self.read_data(y)
        data = self.process_output_data(data)
        return data.T


class CDAQtoNiDAQ(NationalInstrumentsSetup):

    def __init__(self, configs):
        configs['auto_start'] = False
        configs['offset'] = int(configs['sampling_frequency'] * 0.03)  # do not reduce to less than 0.02
        super().__init__(self, configs)
        self.add_channels()

    def get_output(self, y):
        y = y.T
        assert self.configs['shape'] == y.shape[1]
        y = self.synchronise_input_data(y)
        max_attempts = 5
        attempts = 1
        finished = False
        while not finished and (attempts < max_attempts):
            data, finished = self.readout_trial(y)
            attempts += 1

        assert finished, ('Error: output data not same size as input data. Output: ' +
                          str(data.shape[1]) + ' points, input: ' + str(self.configs['shape']) + ' points.')
        return data.T

    def readout_trial(self, y):
        data = self.read_data(y)
        data = self.process_output_data(data)
        data = self.synchronise_output_data(data)
        finished = data.shape[1] == self.configs['shape']
        return data, finished

    def synchronise_input_data(self, y):
        # TODO: Are the following three lines really necessary?
        y = np.asarray(y)
        if len(y.shape) == 1:
            y = y[np.newaxis, :]
        # Append some zeros to the initial signal such that no input data is lost
        # This should be handled with proper synchronization
        y_corr = np.zeros((y.shape[0], y.shape[1] + self.configs['offset']))  # Add 200ms of reaction in terms of zeros
        y_corr[:, self.configs['offset']:] = y[:]
        # TODO: Is this if really necessary?
        if len(y_corr.shape) == 1:
            y_corr = np.concatenate((y_corr[np.newaxis], np.zeros((1, y_corr.shape[1]))))   # Set the trigger
        else:
            y_corr = np.concatenate((y_corr, np.zeros((1, y_corr.shape[1]))))   # Set the trigger
        y_corr[-1, self.configs['offset']] = 1  # Start input data

        return y_corr

    def add_channels(self):
        # Define ao7 as sync signal for the NI 6216 ai0
        self.driver.output_task.ao_channels.add_ao_voltage_chan(self.configs['output_instrument'] + '/ao7', 'ao7', -5, 5)
        self.driver.input_task.ai_channels.add_ai_voltage_chan(self.configs['input_instrument'] + '/ai7')

    def get_output_cut_value(self, read_data):
        cut_value = np.argmax(read_data[-1, :])
        if read_data[-1, cut_value] < 0.05:
            print('Warning: initialize spike not recognized')
        return cut_value

    def synchronise_output_data(self, read_data):
        cut_value = self.get_output_cut_value(read_data)
        return read_data[:-1, cut_value:self.configs['shape'] + cut_value]
