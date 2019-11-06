import numpy as np
import matplotlib.pyplot as plt
from bspyinstr.instruments.instrument_mgr import get_instrument


def sweepgen(v_high, v_low, n_points, direction):
    n_points = n_points / 2

    if direction == 'down':
        input1 = np.linspace(0, v_low, int((n_points * v_low) / (v_low - v_high)))
        input2 = np.linspace(v_low, v_high, n_points)
        input3 = np.linspace(v_high, 0, int((n_points * v_high) / (v_high - v_low)))
    elif direction == 'up':
        input1 = np.linspace(0, v_high, int((n_points * v_high) / (v_high - v_low)))
        input2 = np.linspace(v_high, v_low, n_points)
        input3 = np.linspace(v_low, 0, int((n_points * v_low) / (v_low - v_high)))
    else:
        print('Specify the sweep direction')

    sweep = np.zeros(len(input1) + len(input2) + len(input3))
    sweep[0:len(input1)] = input1
    sweep[len(input1):len(input1) + len(input2)] = input2
    sweep[len(input1) + len(input2):len(input1) + len(input2) + len(input3)] = input3
    return sweep


def meas_ni(v_high, v_low, n_points, direction, instrument, tested_inp, tested_out):
    sweep = sweepgen(v_high, v_low, n_points, direction)
    test_input = np.zeros((len(tested_inp), len(sweep)))
    for i in range(len(tested_inp)):
        test_input[i] = sweep

    test_output = instrument.measure(test_input)
    # measurement_setup.close_tasks()
# Plot the IV curve.
    for i in range(len(tested_out)):
        plt.figure()
        plt.plot(test_input[i], test_output[i])
    plt.show()


def test_cdaq_to_nidaq():
    v_high = 1
    v_low = -1
    direction = 'up'
    n_points = 1000
    tested_inp = [1]
    tested_out = [0]
    configs = {}
    configs['setup_type'] = 'cdaq_to_nidaq'
    configs['shape'] = n_points  # y.shape[1], length of the signal
    configs['input_channels'] = tested_inp
    configs['output_channels'] = tested_out
    configs['sampling_frequency'] = 1000
    instrument = get_instrument(configs)

    try:
        meas_ni(v_high, v_low, n_points, direction, instrument, tested_inp, tested_out)
    finally:
        instrument.close_tasks()


def test_cdaq_to_cdaq():
    v_high = 1
    v_low = -1
    direction = 'up'
    n_points = 1000
    tested_inp = [1, 2]
    tested_out = [1, 2]
    configs = {}
    configs['setup_type'] = 'cdaq_to_cdaq'
    configs['shape'] = n_points  # y.shape[1], length of the signal
    configs['input_channels'] = tested_inp
    configs['output_channels'] = tested_out
    configs['sampling_frequency'] = 1000
    instrument = get_instrument(configs)

    try:
        meas_ni(v_high, v_low, n_points, direction, instrument, tested_inp, tested_out)
    finally:
        instrument.close_tasks()


test_cdaq_to_cdaq()
# meas_ni(1, -1, 1000, 'up', 'nidaq', [1], [0])
