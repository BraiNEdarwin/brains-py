'''

'''
import numpy as np
import math
import time
from bspyproc.processors.hardware import task_mgr
from bspyproc.utils.control import get_control_voltage_indices, merge_inputs_and_control_voltages_in_numpy
import nidaqmx.system.device as device
from multiprocessing import Process

SECURITY_THRESHOLD = 1.5  # Voltage input security threshold


class NationalInstrumentsSetup():

    def __init__(self, configs):
        self.configs = configs
        # self.input_indices = configs['input_indices']
        # self.control_voltage_indices = get_control_voltage_indices(self.input_indices, configs['input_electrode_no'])
        self.driver = task_mgr.get_driver(configs['driver'])
        self.offsetted_shape = configs['shape'] + configs['offset']
        self.ceil = math.ceil((self.offsetted_shape) / self.configs['sampling_frequency']) + 1
        self.driver.init_output(self.configs['input_channels'], self.configs['output_instrument'], self.configs['sampling_frequency'], self.offsetted_shape)
        time.sleep(1)
        self.driver.init_input(self.configs['output_channels'], self.configs['input_instrument'], self.configs['sampling_frequency'], self.offsetted_shape)

    def process_output_data(self, data):
        data = np.asarray(data)
        if len(data.shape) == 1:
            data = data[np.newaxis, :]
        return data * self.configs["amplification"]

    def read_data(self, y):
        p = Process(target=self.download_for, args=(y,))
        p.start()
        p.join()

    def _read_data(self, y):
        '''
            y = It represents the input data as matrix where the shpe is defined by the "number of inputs to the device" times "input points that you want to input to the device".
        '''
        # assert self.offsetted_shape[self.offsetted_shape > SECURITY_THRESHOLD].shape[0] > 0 or self.offsetted_shape[self.offsetted_shape < -SECURITY_THRESHOLD].shape[0] > 0, f"A value is higher/lower than the threshold of +/-{SECURITY_THRESHOLD}. Stopping the program in order to avoid damage to the device."
        self.driver.start_tasks(y, self.configs['auto_start'])
        read_data = self.driver.read(self.offsetted_shape, self.ceil)
        self.driver.stop_tasks()
        return read_data

    def close_tasks(self):
        self.driver.close_tasks()

    def get_amplification_value(self):
        return self.configs["amplification"]

    # def get_output_(self, inputs, control_voltages):
    #     y = merge_inputs_and_control_voltages_in_numpy(inputs, control_voltages, self.input_indices, self.control_voltage_indices)
    #     return self.get_output(y)s

    def get_output(self):
        pass


class CDAQtoCDAQ(NationalInstrumentsSetup):

    def __init__(self, configs):
        configs['auto_start'] = True
        configs['offset'] = 0
        super().__init__(configs)
        self.driver.start_trigger(self.configs['trigger_source'])

    def get_output(self, y):
        y = y.T
        assert self.configs['shape'] == y.shape[1]
        data = self.read_data(y)
        data = self.process_output_data(data)
        return data.T


class CDAQtoNiDAQ(NationalInstrumentsSetup):

    def __init__(self, configs):
        configs['auto_start'] = False

        configs['offset'] = int(configs['sampling_frequency'] * 0.04)  # do not reduce to less than 0.02
        super().__init__(configs)
        self.driver.add_channels(self.configs['output_instrument'], self.configs['input_instrument'])

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

    def get_output_cut_value(self, read_data):
        cut_value = np.argmax(read_data[-1, :])
        if read_data[-1, cut_value] < 0.05:
            print('Warning: initialize spike not recognized')
        return cut_value

    def synchronise_output_data(self, read_data):
        cut_value = self.get_output_cut_value(read_data)
        return read_data[:-1, cut_value:self.configs['shape'] + cut_value]
