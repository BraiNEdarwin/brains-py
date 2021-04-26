"""
Unit tests for iv curves
@author: Michelangelo Barocci and Unai Alegre-Ibarra
"""
import matplotlib.pyplot as plt
from brainspy.utils.io import load_configs
from brainspy.utils.manager import get_driver
import numpy as np


class IVMeasurement():

    def __init__(self, configs):
        self.configs = configs
        self.input_signal = self.configs['input_signal']
        self.index_prog = {}
        self.index_prog["all"] = 0
        for dev in self.configs['devices']:
            self.index_prog[dev] = 0

    def run_test(self):

        # save(mode='configs', path=self.configs['results_base_dir'], filename='test_configs.json', overwrite=self.configs['overwrite_results'], data=self.configs)

        self.driver = get_driver(self.configs['driver'])
        experiments = ["IV1", "IV2", "IV3", "IV4", "IV5", "IV6", "IV7"]
        self.devices_in_experiments = {}
        output = {}
        output_array = []

        for exp in experiments:
            output[exp] = {}
            self.devices_in_experiments[exp] = self.configs['devices'].copy()
            output_array = self.driver.forward_numpy(self.create_input_arrays())

            for i, dev in enumerate(self.configs['devices']):
                output[exp][dev] = output_array.T[i, :]

        self.iv_plot(configs, output)
        self.driver.close_tasks()

    def create_input_arrays(self):

        inputs_dict = {}
        inputs_array = []

        for dev in self.configs['devices']:

            inputs_dict[dev] = np.zeros((self.configs["driver"]['instruments_setup'][dev]['activation_channel_mask'].count(1), self.configs['shape']))  # creates a zeros array for each '1' in the mask entry

            if self.configs["driver"]['instruments_setup'][dev]['activation_channel_mask'][self.index_prog["all"]] == 1:
                inputs_dict[dev][self.index_prog[dev], :] = self.gen_input_wfrm()
                self.index_prog[dev] += 1

            else:
                self.devices_in_experiments["IV" + str(self.index_prog["all"] + 1)].remove(dev)

            inputs_array.extend(inputs_dict[dev])

        inputs_array = np.array(inputs_array)
        self.index_prog["all"] += 1

        return inputs_array.T

    def gen_input_wfrm(self):

        def generate_sawtooth(v_high, v_low, n_points, direction):

            n_points = n_points / 2

            if direction == "up":
                Input1 = np.linspace(0, v_low, int((n_points * v_low) / (v_low - v_high)))
                Input2 = np.linspace(v_low, v_high, int(n_points))
                Input3 = np.linspace(v_high, 0, int((n_points * v_high) / (v_high - v_low)))
            elif direction == "down":
                Input1 = np.linspace(0, v_high, int((n_points * v_high) / (v_high - v_low)))
                Input2 = np.linspace(v_high, v_low, int(n_points))
                Input3 = np.linspace(v_low, 0, int((n_points * v_low) / (v_low - v_high)))
            else:
                print('Specify the sweep direction')

            Input = np.zeros(len(Input1) + len(Input2) + len(Input3))
            Input[0:len(Input1)] = Input1
            Input[len(Input1):len(Input1) + len(Input2)] = Input2
            Input[len(Input1) + len(Input2):len(Input1) + len(Input2) + len(Input3)] = Input3

            return Input

        def generate_sinewave(n, fs, amplitude, phase=0):
            '''
            Generates a sine wave that can be used for the input data.
            freq:       Frequencies of the inputs in an one-dimensional array
            t:          The datapoint(s) index where to generate a sine value (1D array when multiple datapoints are used)
            amplitude:  Amplitude of the sine wave (Vmax in this case)
            fs:         Sample frequency of the device
            phase:      (Optional) phase offset at t=0
            '''
            freq = fs / n
            points = np.linspace(0, 1 / freq, n)
            phases = points * 2 * np.pi * freq

            return np.sin(phases + phase) * amplitude

        if self.input_signal['input_signal_type'] == 'sawtooth':
            input_data = generate_sawtooth(self.input_signal['voltage_range'][1], self.input_signal['voltage_range'][0], self.configs['shape'], self.input_signal['direction'])
        elif self.input_signal['input_signal_type'] == 'sine':
            input_data = generate_sinewave(self.configs['shape'], self.configs["driver"]['sampling_frequency'], self.input_signal['V_high'])
            input_data[-1] = 0
        else:
            print("Specify input_signal type")

        return input_data

    def plot(self, x, y):
        for i in range(np.shape(y)[1]):
            plt.figure()
            plt.plot(x)
            plt.plot(y)
            plt.show()

    def iv_plot(self, configs, output):

        xaxis = self.gen_input_wfrm()
        devlist = configs['driver']['instruments_setup']  # get_default_brains_setup_dict()
        ylabeldist = -5

        for dev in self.configs['devices']:
            fig, axs = plt.subplots(2, 4)
            # plt.grid(True)
            fig.suptitle('Device ' + dev + ' - Input voltage vs Output current')
            for i in range(2):
                for j in range(4):
                    exp = "IV" + str(j + i * 4 + 1)
                    if j + i * 4 < 7:
                        if self.configs["driver"]['instruments_setup'][dev]["activation_channel_mask"][j + i * 4] == 1:
                            axs[i, j].plot(xaxis, output[exp][dev])
                            axs[i, j].set_ylabel('output (nA)', labelpad=ylabeldist)
                            axs[i, j].set_xlabel('input (V)', labelpad=1)
                            axs[i, j].xaxis.grid(True)
                            axs[i, j].yaxis.grid(True)
                        else:
                            axs[i, j].plot(xaxis, xaxis * 0)
                            axs[i, j].set_xlabel('Channel Masked')
                        axs[i, j].set_title(devlist[dev]["activation_channels"][j + i * 4])
                    else:
                        axs[i, j].plot(xaxis)
                        axs[i, j].yaxis.tick_right()
                        axs[i, j].yaxis.set_label_position("right")
                        axs[i, j].set_ylabel('input (V)')
                        axs[i, j].set_xlabel('points', labelpad=1)
                        axs[i, j].set_title("Input input_signal")
                        axs[i, j].xaxis.grid(True)
                        axs[i, j].yaxis.grid(True)
        plt.show()


if __name__ == '__main__':

    from brainspy.utils.io import load_configs
    configs = {}
    configs['results_base_dir'] = 'tmp/tests/iv'
    configs['show_plots'] = True
    configs['devices'] = ['A', 'B', 'C', "D", 'E']
    configs['shape'] = 500  # length of the experiment
    configs['input_signal'] = {}
    configs['input_signal']['voltage_range'] = [-0.75,0.75]
    configs['input_signal']['input_signal_type'] = 'sawtooth' # Type of signal to be created in the input. It can either be 'sine' or 'sawtooth'
    configs['input_signal']['time_in_seconds'] = 5 # time_in_seconds in seconds
    configs['input_signal']['direction'] = 'up'

    configs['driver'] = load_configs('C:/Users/Unai/Documents/programming/brains-py/tests/hardware/brains_ivcurve.yaml')

    test = IVMeasurement(configs)
    test.run_test()
    #suite = unittest.TestSuite()
    #suite.addTest(IVtest(configs))
    #unittest.TextTestRunner().run(suite)
