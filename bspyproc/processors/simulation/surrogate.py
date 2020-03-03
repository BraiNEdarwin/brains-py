""" The accelerator class enables to statically access the accelerator
(CUDA or CPU) that is used in the computer. The aim is to support both platforms seemlessly. """

import torch
from bspyproc.processors.simulation.network import NeuralNetworkModel
from bspyproc.utils.pytorch import TorchUtils
from bspyproc.utils.control import merge_inputs_and_control_voltages_in_numpy, get_control_voltage_indices


class SurrogateModel(NeuralNetworkModel):
    """
        The TorchModel class is used to manage together a torch model and its state dictionary. The usage is expected to be as follows
        mymodel = TorchModel()
        mymodel.load_model('my_path/my_model.pt')
        mymodel.model
    """
# TODO: Automatically register the data type according to the configurations of the amplification variable of the  info dictionary

    def __init__(self, configs):
        self.load_model(configs)
        if 'input_indices' in configs and 'input_electrode_no' in configs:
            self.input_indices = configs['input_indices']
            self.control_voltage_indices = get_control_voltage_indices(self.input_indices, configs['input_electrode_no'])
        else:
            print('Warning: Input indices and control voltage indices have not been defined.')

    def load_file(self, data_dir, file_type):
        if file_type == 'pt':
            state_dict = torch.load(data_dir, map_location=TorchUtils.get_accelerator_type())
            info = state_dict['info']
            del state_dict['info']
            info['smg_configs'] = self._info_consistency_check(info['smg_configs'])
            if 'amplification' not in info['data_info']['processor'].keys():
                info['data_info']['processor']['amplification'] = 1
        elif file_type == 'json':
            state_dict = None
            # TODO: Implement loading from a json file
            raise NotImplementedError(f"Loading file from a json file in TorchModel has not been implemented yet. ")
            # info = model_info loaded from a json file
        return info, state_dict

    def init_noise_configs(self):
        if 'noise' in self.configs:
            print(f"The model has a gaussian noise based on a MSE of {torch.sqrt(torch.tensor([self.configs['noise']]))}")
            self.error = TorchUtils.format_tensor(torch.sqrt(torch.tensor([self.configs['noise']])))
            self.forward_processed = self.forward_amplification_and_noise
        else:
            print(f"The model has been initialised without noise.")
            self.error = TorchUtils.format_tensor(torch.tensor([0]))
            self.forward_processed = self.forward_amplification

    def load_model(self, configs):
        """Loads a pytorch model from a directory string."""
        self.info, state_dict = self.load_file(configs['torch_model_dict'], 'pt')
        if 'smg_configs' in self.info.keys():
            model_dict = self.info['smg_configs']['processor']
        else:
            model_dict = self.info
        super().__init__(model_dict)
        self.configs = configs
        self.load_state_dict(state_dict)
        self.init_max_and_min_values()
        self.amplification = TorchUtils.get_tensor_from_list(self.info['data_info']['processor']['amplification'])
        self.init_noise_configs()

    def init_max_and_min_values(self):
        self.offset = TorchUtils.get_tensor_from_list(self.info['data_info']['input_data']['offset'])
        self.amplitude = TorchUtils.get_tensor_from_list(self.info['data_info']['input_data']['amplitude'])
        self.min_voltage = self.offset - self.amplitude
        self.max_voltage = self.offset + self.amplitude

    def reset(self):
        print("Warning: Reset function in Surrogate Model not implemented.")

    def get_output(self, input_matrix):
        with torch.no_grad():
            inputs_torch = TorchUtils.get_tensor_from_numpy(input_matrix)
            output = self.forward_processed(inputs_torch)
        return TorchUtils.get_numpy_from_tensor(output)

    def forward_amplification(self, x):
        return self.model(x) * self.amplification

    def forward_amplification_and_noise(self, x):
        output = self.forward_amplification(x)
        noise = self.error * TorchUtils.format_tensor(torch.randn(output.shape))
        return output + noise

    def forward(self, x):
        return self.forward_processed(x)

    def get_amplification_value(self):
        return self.info['data_info']['processor']['amplification']
