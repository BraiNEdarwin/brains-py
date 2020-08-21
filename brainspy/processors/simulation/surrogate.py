""" The accelerator class enables to statically access the accelerator
(CUDA or CPU) that is used in the computer. The aim is to support both platforms seemlessly. """

import torch
from torch import nn

from bspyproc.processors.simulation.network import NeuralNetworkModel
from bspyproc.utils.pytorch import TorchUtils
from bspyproc.utils.loader import load_file
from bspyproc.processors.simulation.noise.noise import get_noise


class SurrogateModel(nn.Module):
    """
        The TorchModel class is used to manage together a torch model and its state dictionary. The usage is expected to be as follows
        mymodel = TorchModel()
        mymodel.load_model('my_path/my_model.pt')
        mymodel.model
    """
# TODO: Automatically register the data type according to the configurations of the amplification variable of the  info dictionary

    def __init__(self, configs):
        super().__init__()
        self._load(configs)
        self._init_voltage_ranges()
        self.amplification = TorchUtils.get_tensor_from_list(self.info['data_info']['processor']['amplification'])
        self.clipping_value = TorchUtils.get_tensor_from_list(self.info['data_info']['clipping_value'])
        self.noise = get_noise(configs)

    def _load(self, configs):
        """Loads a pytorch model from a directory string."""
        self.configs = configs
        self.info, state_dict = load_file(configs['torch_model_dict'], 'pt')
        self.model = NeuralNetworkModel(self.info['smg_configs']['processor'])
        self.load_state_dict(state_dict)

    def _init_voltage_ranges(self):
        offset = TorchUtils.get_tensor_from_list(self.info['data_info']['input_data']['offset'])
        amplitude = TorchUtils.get_tensor_from_list(self.info['data_info']['input_data']['amplitude'])
        min_voltage = (offset - amplitude).unsqueeze(dim=1)
        max_voltage = (offset + amplitude).unsqueeze(dim=1)
        self.voltage_ranges = torch.cat((min_voltage, max_voltage), dim=1)

    def forward(self, x):
        return self.noise(self.model(x) * self.amplification)

    def forward_numpy(self, input_matrix):
        with torch.no_grad():
            inputs_torch = TorchUtils.get_tensor_from_numpy(input_matrix)
            output = self.forward(inputs_torch)
        return TorchUtils.get_numpy_from_tensor(output)

    def reset(self):
        print("Warning: Reset function in Surrogate Model not implemented.")
        pass
