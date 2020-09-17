"""
Unit tests for iv curves
@author: Michelangelo Barocci and Unai Alegre-Ibarra
"""
from brainspy.processors.hardware.drivers.ni.defaults import get_default_brains_setup_dict
import matplotlib.pyplot as plt
from bspyalgo.utils.io import load_configs
from brainspy.utils.manager import get_driver
import numpy as np
import unittest2 as unittest


class IVtest(unittest.TestCase):

    def __init__(self, configs):
        self.configs = configs
        # self.processor = self.configs['processor']
        self.waveform = self.configs['waveform']
        self.index_prog = {}
        self.index_prog["all"] = 0
        for dev in self.configs["processor"]['devices']:
            self.index_prog[dev] = 0

    def runTest(self):

        # save(mode='configs', path=self.configs['results_base_dir'], filename='test_configs.json', overwrite=self.configs['overwrite_results'], data=self.configs)

        self.processor = get_driver(self.configs['processor'])
        experiments = ["IV1", "IV2", "IV3", "IV4", "IV5", "IV6", "IV7"]
        self.devices_in_experiments = {}
        output = {}
        output_array = []

        for exp in experiments:
            output[exp] = {}
            self.devices_in_experiments[exp] = self.configs["processor"]['devices'].copy()
            output_array = self.processor.forward_numpy(IVtest.create_input_arrays(self))

            for i, dev in enumerate(self.configs["processor"]['devices']):
                output[exp][dev] = output_array[i, :]

        self.iv_plot(output)
        #self.plot(IVtest.gen_input_wfrm(self), output["IV7"]["E"])

        # input_data = np.zeros((2,1000))
        # input_data[0,:] = IVtest.gen_input_wfrm(self)
        # input_data[1,:] = IVtest.gen_input_wfrm(self)
        # output_array = self.processor.get_output(input_data)
        # self.plot(IVtest.gen_input_wfrm(self), output_array)

    def create_input_arrays(self):

        inputs_dict = {}
        inputs_array = []

        for dev in self.configs["processor"]['devices']:

            inputs_dict[dev] = np.zeros((self.configs["processor"]['activation_channel_mask'][dev].count(1), self.configs["processor"]['shape']))  # creates a zeros array for each '1' in the mask entry

            if self.configs["processor"]['activation_channel_mask'][dev][self.index_prog["all"]] == 1:
                inputs_dict[dev][self.index_prog[dev], :] = IVtest.gen_input_wfrm(self)
                self.index_prog[dev] += 1

            else:
                self.devices_in_experiments["IV" + str(self.index_prog["all"] + 1)].remove(dev)

            inputs_array.extend(inputs_dict[dev])

        inputs_array = np.array(inputs_array)
        self.index_prog["all"] += 1

        return inputs_array

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

        if self.waveform['input_type'] == 'sawtooth':
            input_data = generate_sawtooth(self.waveform['V_high'], self.waveform['V_low'], self.configs["processor"]['shape'], self.waveform['direction'])
        elif self.waveform['input_type'] == 'sine':
            input_data = generate_sinewave(self.configs["processor"]['shape'], self.configs["processor"]['sampling_frequency'], self.waveform['V_high'])
            input_data[-1] = 0
        else:
            print("Specify waveform type")

        return input_data

    # def close_test(self, results):
        # self.excel_file.close_file()
        # self.results = results
        # return results

    def plot(self, x, y):
        for i in range(np.shape(y)[1]):
            plt.figure()
            plt.plot(x)
            plt.plot(y)
            plt.show()

        # fig, ax1 = plt.subplots()

        # color = 'tab:blue'
        # ax1.set_ylabel('input (V)', color=color)  # we already handled the x-label with ax1
        # ax1.plot(x, color=color)
        # ax1.tick_params(axis='y', labelcolor=color)

        # color = 'tab:orange'
        # ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        # ax2.set_xlabel('time (s)')
        # ax2.set_ylabel('output (nA)', color=color)
        # ax2.plot(y, color=color)
        # ax2.tick_params(axis='y', labelcolor=color)

        # fig.tight_layout()  # otherwise the right y-label is slightly clipped
        # plt.show()

    def iv_plot(self, output):

        xaxis = IVtest.gen_input_wfrm(self)
        devlist = get_default_brains_setup_dict()
        ylabeldist = -5

        for dev in self.configs["processor"]['devices']:
            fig, axs = plt.subplots(2, 4)
            # plt.grid(True)
            fig.suptitle('Device ' + dev + ' - Input voltage vs Output current')
            for i in range(2):
                for j in range(4):
                    exp = "IV" + str(j + i * 4 + 1)
                    if j + i * 4 < 7:
                        if self.configs["processor"]["activation_channel_mask"][dev][j + i * 4] == 1:
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
                        axs[i, j].set_title("Input Waveform")
                        axs[i, j].xaxis.grid(True)
                        axs[i, j].yaxis.grid(True)
        plt.show()


if __name__ == '__main__':
    IV_test_configs = load_configs('configs/IV5_template.json')

    suite = unittest.TestSuite()
    suite.addTest(IVtest(IV_test_configs))
    unittest.TextTestRunner().run(suite)
