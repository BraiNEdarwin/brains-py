
import numpy as np
from bspyinstr.utils.waveform import generate_waveform


class WaveformManager:

    def __init__(self, configs):
        self.configs = configs

    def waveform(self, data):
        data_wfrm = generate_waveform(data, self.configs['lengths'], slopes=self.configs['slopes'])
        return np.asarray(data_wfrm)

    def input_waveform(self, inputs):
        nr_inp = len(inputs)
        print(f'Input is {nr_inp} dimensional')
        inp_list = [self.waveform(inputs[i]) for i in range(nr_inp)]
        inputs_wvfrm = np.asarray(inp_list)

        samples = inputs_wvfrm.shape[-1]
        print(f'Input signals have length {samples}')

        w_ampl = [1, 0] * len(inputs[0])
        if(type(self.configs['lengths']) is int and type(self.configs['slopes']) is int):
            w_lengths = [self.configs['lengths'], self.configs['slopes']] * len(inputs[0])
        else:
            w_lengths = [self.configs['lengths'][0], self.configs['slopes'][0]] * len(inputs[0])
        weight_wvfrm = generate_waveform(w_ampl, w_lengths)
        bool_weights = [x == 1 for x in weight_wvfrm[:samples]]

        return inputs_wvfrm, bool_weights
