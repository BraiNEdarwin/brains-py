import numpy as np
from scipy import signal


def generate_triangle(self, freq, t, amplitude, fs, phase=np.zeros(7)):
    '''
    Generates a triangle wave form that can be used for the input data.
    freq:       Frequencies of the inputs in an one-dimensional array
    t:          The datapoint(s) index where to generate a sine value (1D array when multiple datapoints are used)
    amplitude:  Amplitude of the sine wave (Vmax in this case)
    fs:         Sample frequency of the device
    phase:      (Optional) phase offset at t=0
    '''
    # There is an additional + np.pi/2 to make sure that if phase = 0. the inputs start at 0V

    return signal.sawtooth((2 * np.pi * freq[:, np.newaxis] * t) / fs + phase[:, np.newaxis] + np.pi / 2, 0.5) * amplitude[:, np.newaxis]


def generate_sinewave(self, freq, t, amplitude, fs, phase=np.zeros(7)):
    '''
    Generates a sine wave that can be used for the input data.
    freq:       Frequencies of the inputs in an one-dimensional array
    t:          The datapoint(s) index where to generate a sine value (1D array when multiple datapoints are used)
    amplitude:  Amplitude of the sine wave (Vmax in this case)
    fs:         Sample frequency of the device
    phase:      (Optional) phase offset at t=0
    '''

    return np.sin((2 * np.pi * freq[:, np.newaxis] * t) / fs + phase[:, np.newaxis]) * amplitude[:, np.newaxis]


def get_control_voltage_indices(input_indices, length):
    processor_input = np.arange(length)
    return np.setdiff1d(processor_input, np.array(input_indices))


def merge_inputs_and_control_voltages(inputs, control_voltages, input_indices, control_voltage_indices):
    result = np.empty((inputs.shape[0], len(input_indices) + len(control_voltage_indices)))
    result[:, input_indices] = inputs
    result[:, control_voltage_indices] = control_voltages

    return result
