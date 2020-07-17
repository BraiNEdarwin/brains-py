""" The accelerator class enables to statically access the accelerator
(CUDA or CPU) that is used in the computer. The aim is to support both platforms seemlessly. """

import torch
import torch.nn
from bspyproc.processors.simulation.network import NeuralNetworkModel
from bspyproc.utils.pytorch import TorchUtils
from bspyproc.utils.control import get_control_voltage_indices
from utils.dictionaries import info_consistency_check
from utils.waveform import WaveformManager


class HardwareProcessor(nn.Module):
    """
        The TorchModel class is used to manage together a torch model and its state dictionary. The usage is expected to be as follows
        mymodel = TorchModel()
        mymodel.load_model('my_path/my_model.pt')
        mymodel.model
    """
# TODO: Automatically register the data type according to the configurations of the amplification variable of the  info dictionary

    def __init__(self, configs):
        super().__init__()
        self.load_model(configs)
        self._init_voltage_range()
        self.waveform_mgr = WaveformManager(configs)

    def load(self, configs):
        pass

    def _init_voltage_range(self):
        offset = TorchUtils.get_tensor_from_list(self.info['data_info']['input_data']['offset'])
        amplitude = TorchUtils.get_tensor_from_list(self.info['data_info']['input_data']['amplitude'])
        self.min_voltage = offset - amplitude
        self.max_voltage = offset + amplitude

    def forward(self, x):
        return (self.model(x) * self.amplification) + (self.error * TorchUtils.format_tensor(torch.randn(output.shape)))

    def forward_numpy(self, input_matrix):
        with torch.no_grad():
            inputs_torch = TorchUtils.get_tensor_from_numpy(input_matrix)
            output = self.forward(inputs_torch)
        return TorchUtils.get_numpy_from_tensor(output)

    def reset(self):
        print("Warning: Reset function in Surrogate Model not implemented.")
        self.model.reset()

    def close(self):
        pass
