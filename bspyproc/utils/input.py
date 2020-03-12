import numpy as np
from scipy import signal


def generate_triangle(freq, t, amplitude, fs, phase=np.zeros(7)):
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


def generate_sinewave(freq, t, amplitude, fs, phase=np.zeros(7)):
    '''
    Generates a sine wave that can be used for the input data.
    freq:       Frequencies of the inputs in an one-dimensional array
    t:          The datapoint(s) index where to generate a sine value (1D array when multiple datapoints are used)
    amplitude:  Amplitude of the sine wave (Vmax in this case)
    fs:         Sample frequency of the device
    phase:      (Optional) phase offset at t=0
    '''

    return np.sin((2 * np.pi * freq[:, np.newaxis] * t) / fs + phase[:, np.newaxis]) * amplitude[:, np.newaxis]


def normalise(x, eps=1e-5):
    return ((x - np.mean(x)) / (np.sqrt(np.var(x) + eps)))


def map_to_voltage(x, v_min, v_max):
    a = ((v_min - v_max) / (x.min() - x.max()))
    b = v_max - a * x.max()
    return (a * x) + b

def get_map_to_voltage_vars(v_min, v_max, x=np.array([-1,1])):
    scale = ((v_min - v_max) / (x.min() - x.max()))
    offset = v_max - scale * x.max()
    return scale, offset
