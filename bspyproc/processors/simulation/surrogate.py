""" The accelerator class enables to statically access the accelerator
(CUDA or CPU) that is used in the computer. The aim is to support both platforms seemlessly. """

import torch
import torch.nn
from bspyproc.processors.simulation.network import NeuralNetworkModel
from bspyproc.utils.pytorch import TorchUtils
from bspyproc.utils.control import get_control_voltage_indices
from utils.dictionaries import info_consistency_check


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
        self.load_model(configs)
        self._init_voltage_range()
        self.amplification = TorchUtils.get_tensor_from_list(self.info['data_info']['processor']['amplification'])
        self._init_noise_configs()

    def load_model(self, configs):
        """Loads a pytorch model from a directory string."""
        self.configs = configs
        self.info, state_dict = self._load_file(configs['torch_model_dict'], 'pt')
        self.model = NeuralNetworkModel(self.info['smg_configs']['processor'])
        self.load_state_dict(state_dict)
        self._init_info()

    def _load_file(self, data_dir, file_type):
        if file_type == 'pt':
            state_dict = torch.load(data_dir, map_location=TorchUtils.get_accelerator_type())
            info = state_dict['info']
            del state_dict['info']
            info['smg_configs'] = info_consistency_check(info['smg_configs'])
            if 'amplification' not in info['data_info']['processor'].keys():
                info['data_info']['processor']['amplification'] = 1
        elif file_type == 'json':
            state_dict = None
            # TODO: Implement loading from a json file
            raise NotImplementedError(f"Loading file from a json file in TorchModel has not been implemented yet. ")
            # info = model_info loaded from a json file
        return info, state_dict

    def _init_voltage_range(self):
        offset = TorchUtils.get_tensor_from_list(self.info['data_info']['input_data']['offset'])
        amplitude = TorchUtils.get_tensor_from_list(self.info['data_info']['input_data']['amplitude'])
        self.min_voltage = offset - amplitude
        self.max_voltage = offset + amplitude

    def _init_noise_configs(self):
        if 'noise' in self.configs:
            print(f"The model has a gaussian noise based on a MSE of {torch.tensor([self.configs['noise']])}")
            self.error = torch.sqrt(TorchUtils.format_tensor(torch.tensor([self.configs['noise']])))
        else:
            print(f"The model has been initialised without noise.")
            self.error = TorchUtils.format_tensor(torch.tensor([0]))

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
