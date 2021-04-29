"""
Unit tests for iv curves
@author: Michelangelo Barocci and Unai Alegre-Ibarra
"""
import matplotlib.pyplot as plt
from brainspy.utils.io import load_configs
from brainspy.utils.manager import get_driver
import numpy as np
import math


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
        input_arrays = []
        for i, exp in enumerate(experiments):
            output[exp] = {}
            input_array = self.create_input_arrays()
            self.devices_in_experiments[exp] = self.configs['devices'].copy()
            output_array = self.driver.forward_numpy(input_array)
            input_arrays.append(input_array[:, i])
            for j, dev in enumerate(self.configs['devices']):
                output[exp][dev] = output_array[:, j]
        self.driver.close_tasks()
        self.iv_plot(configs, np.array(input_arrays).T, output)

    # def create_input_arrays(self):

    #     inputs_dict = {}
    #     inputs_array = []

    #     for dev in self.configs['devices']:
    #         current_mask = self.configs["driver"]['instruments_setup'][dev][
    #             'activation_channel_mask']
    #         input_wfrm = self.gen_input_wfrm(
    #             self.configs["driver"]['instruments_setup'][dev]
    #             ['voltage_ranges'][self.index_prog[dev]])
    #         inputs_dict[dev] = np.zeros_like(
    #             input_wfrm[:, np.array(current_mask) == 1]
    #         )  # creates a zeros array for each '1' in the mask entry

    #         if current_mask[self.index_prog["all"]] == 1:
    #             inputs_dict[dev] = input_wfrm.copy(
    #             )  #inputs_dict[dev][self.index_prog[dev], :] = input_wfrm.copy()
    #             self.index_prog[dev] += 1

    #         else:
    #             self.devices_in_experiments["IV" + str(self.index_prog["all"] +
    #                                                    1)].remove(dev)

    #         inputs_array.append(inputs_dict[dev])

    #     #inputs_array = np.array(inputs_array)
    #     self.index_prog["all"] += 1

    #     return np.concatenate(inputs_array, axis=1)

    def create_input_arrays(self):

        inputs_dict = {}
        inputs_array = []

        for dev in self.configs['devices']:

            inputs_dict[dev] = np.zeros(
                (self.configs["driver"]['instruments_setup'][dev]
                 ['activation_channel_mask'].count(1), self.configs['shape']
                 ))  # creates a zeros array for each '1' in the mask entry

            if self.configs["driver"]['instruments_setup'][dev][
                    'activation_channel_mask'][self.index_prog["all"]] == 1:
                inputs_dict[dev][
                    self.index_prog[dev], :] = self.gen_input_wfrm(
                        self.configs["driver"]['instruments_setup'][dev]
                        ['voltage_ranges'][self.index_prog[dev]])
                self.index_prog[dev] += 1

            else:
                self.devices_in_experiments["IV" + str(self.index_prog["all"] +
                                                       1)].remove(dev)

            inputs_array.extend(inputs_dict[dev])

        inputs_array = np.array(inputs_array)
        self.index_prog["all"] += 1

        return inputs_array.T

    def gen_input_wfrm(self, input_range):
        if self.input_signal['input_signal_type'] == 'sawtooth':
            input_data = generate_sawtooth(input_range, self.configs['shape'],
                                           self.input_signal['direction'])
        elif self.input_signal['input_signal_type'] == 'sine':
            input_data = generate_sinewave(
                self.configs['shape'],
                self.configs["driver"]['sampling_frequency'],
                input_range[1])  # Max from the input range
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

    def iv_plot(self, configs, input_waveform, output):

        #xaxis = self.gen_input_wfrm()
        devlist = configs['driver'][
            'instruments_setup']  # get_default_brains_setup_dict()
        ylabeldist = -5
        electrode_id = 0
        for k, dev in enumerate(self.configs['devices']):
            fig, axs = plt.subplots(2, 4)
            # plt.grid(True)
            fig.suptitle('Device ' + dev +
                         ' - Input voltage vs Output current')
            for i in range(2):
                for j in range(4):
                    current_electrode = j + i * 4
                    exp = "IV" + str(current_electrode + 1)
                    if current_electrode < 7:
                        if self.configs["driver"]['instruments_setup'][dev][
                                "activation_channel_mask"][
                                    current_electrode] == 1:
                            axs[i, j].plot(input_waveform[:, electrode_id],
                                           output[exp][dev])
                            axs[i, j].set_ylabel('output (nA)',
                                                 labelpad=ylabeldist)
                            axs[i, j].set_xlabel('input (V)', labelpad=1)
                            axs[i, j].xaxis.grid(True)
                            axs[i, j].yaxis.grid(True)
                        else:
                            axs[i, j].plot(input_waveform[:, electrode_id],
                                           input_waveform[:, electrode_id] * 0)
                            axs[i, j].set_xlabel('Channel Masked')
                        axs[i, j].set_title(devlist[dev]["activation_channels"]
                                            [current_electrode])
                        electrode_id += 1
                    else:
                        for z in range(7):
                            axs[i, j].plot(input_waveform[:, (k * 7) + z],
                                           label="IV" + str(z + 1))
                        #axs[i, j].yaxis.tick_right()
                        #axs[i, j].yaxis.set_label_position("right")
                        axs[i, j].set_ylabel('input (V)')
                        axs[i, j].set_xlabel('points', labelpad=1)
                        axs[i, j].set_title("Input input_signal")
                        axs[i, j].xaxis.grid(True)
                        axs[i, j].yaxis.grid(True)
                        axs[i, j].legend()
        plt.subplots_adjust(hspace=0.3, wspace=0.35)
        plt.show()


def generate_sawtooth(input_range, n_points, direction):

    n_points = n_points / 2

    if direction == "up":
        Input1 = np.linspace(
            0, input_range[0],
            int((n_points * input_range[0]) /
                (input_range[0] - input_range[1])))
        Input2 = np.linspace(input_range[0], input_range[1], int(n_points))
        Input3 = np.linspace(
            input_range[1], 0,
            int((n_points * input_range[1]) /
                (input_range[1] - input_range[0])))
    elif direction == "down":
        Input1 = np.linspace(
            0, input_range[1],
            int((n_points * input_range[1]) /
                (input_range[1] - input_range[0])))
        Input2 = np.linspace(input_range[1], input_range[0], int(n_points))
        Input3 = np.linspace(
            input_range[0], 0,
            int((n_points * input_range[0]) /
                (input_range[0] - input_range[1])))
    else:
        print('Specify the sweep direction')
    result = np.concatenate((Input1, Input2, Input3))
    if not (result.shape[0] == int(n_points * 2)):
        result = np.concatenate((result, np.array([0])))
    return result


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


if __name__ == '__main__':

    from brainspy.utils.io import load_configs
    configs = {}
    configs['results_base_dir'] = 'tmp/tests/iv'
    configs['show_plots'] = True
    configs['devices'] = ["D"]
    # configs['devices'] = [
    #     "A", "B", "C", "D", "E"
    # ]  #["D"]  # To remove devices from this list, set the mask to zero first in the configs.
    configs['shape'] = 500  # length of the experiment
    configs['input_signal'] = {}
    #configs['input_signal']['voltage_range'] = [1.2, 0.5]
    configs['input_signal'][
        'input_signal_type'] = 'sine'  # Type of signal to be created in the input. It can either be 'sine' or 'sawtooth'
    configs['input_signal'][
        'time_in_seconds'] = 5  # time_in_seconds in seconds
    configs['input_signal']['direction'] = 'up'

    configs['driver'] = load_configs('tests/hardware/brains_ivcurve.yaml')

    test = IVMeasurement(configs)
    test.run_test()
    #suite = unittest.TestSuite()
    #suite.addTest(IVtest(configs))
    #unittest.TextTestRunner().run(suite)
