""" The accelerator class enables to statically access the accelerator
(CUDA or CPU) that is used in the computer. The aim is to support both platforms seemlessly. """

import torch
import torch.nn as nn
from bspyproc.utils.pytorch import TorchUtils
from bspyproc.utils.control import merge_inputs_and_control_voltages, get_control_voltage_indices


class TorchModel(nn.Module):
    """
        The TorchModel class is used to manage together a torch model and its state dictionary. The usage is expected to be as follows
        mymodel = TorchModel()
        mymodel.load_model('my_path/my_model.pt')
        mymodel.model
    """
# TODO: Automatically register the data type according to the configurations of the amplification variable of the  info dictionary

    def __init__(self, configs):
        super().__init__()

        self.configs = configs
        if type(configs['torch_model_dict']) is str:
            self.load_model(configs['torch_model_dict'])
        elif type(configs['torch_model_dict']) is dict:
            self.build_model(configs['torch_model_dict'])
        if 'input_indices' in configs and 'input_electrode_no' in configs:
            self.input_indices = configs['input_indices']
            self.control_voltage_indices = get_control_voltage_indices(self.input_indices, configs['input_electrode_no'])
        else:
            print('Warning: Input indices and control voltage indices have not been defined.')
        if TorchUtils.get_accelerator_type() == torch.device('cuda'):
            self.model.cuda()

    def load_file(self, data_dir, file_type):
        if file_type == 'pt':
            state_dict = torch.load(data_dir, map_location=TorchUtils.get_accelerator_type())
            info = state_dict['info']
            info['smg_configs'] = self._info_consistency_check(info['smg_configs'])
            del state_dict['info']
            if 'amplification' not in info['data_info']['processor'].keys():
                info['data_info']['processor']['amplification'] = 1
        elif file_type == 'json':
            state_dict = None
            # TODO: Implement loading from a json file
            raise NotImplementedError(f"Loading file from a json file in TorchModel has not been implemented yet. ")
            # info = model_info loaded from a json file
        return info, state_dict

    def load_model(self, data_dir):
        """Loads a pytorch model from a directory string."""
        self.info, state_dict = self.load_file(data_dir, 'pt')
        if 'smg_configs' in self.info.keys():
            model_dict = self.info['smg_configs']['processor']['torch_model_dict']
        else:
            model_dict = self.info
        self.build_model(model_dict)
        self.model.load_state_dict(state_dict)
        self.amplification = TorchUtils.get_tensor_from_list(self.info['data_info']['processor']['amplification'])
        self.offset = TorchUtils.get_tensor_from_list(self.info['data_info']['input_data']['offset'])
        self.amplitude = TorchUtils.get_tensor_from_list(self.info['data_info']['input_data']['amplitude'])
        self.min_voltage = self.offset - self.amplitude
        self.max_voltage = self.offset + self.amplitude

    def build_model(self, model_info):

        hidden_sizes = model_info['hidden_sizes']
        input_layer = nn.Linear(model_info['D_in'], hidden_sizes[0])
        activ_function = self._get_activation(model_info['activation'])
        output_layer = nn.Linear(hidden_sizes[-1], model_info['D_out'])
        modules = [input_layer, activ_function]

        hidden_layers = zip(hidden_sizes[: -1], hidden_sizes[1:])
        for h_1, h_2 in hidden_layers:
            hidden_layer = nn.Linear(h_1, h_2)
            modules.append(hidden_layer)
            modules.append(activ_function)

        modules.append(output_layer)
        self.model = nn.Sequential(*modules)

        print('Model built with the following modules: \n', modules)

    def get_output(self, input_matrix):
        with torch.no_grad():
            inputs_torch = TorchUtils.get_tensor_from_numpy(input_matrix)
            output = self.forward(inputs_torch)
        return TorchUtils.get_numpy_from_tensor(output) * self.amplification

    def get_output_(self, inputs, control_voltages):
        y = merge_inputs_and_control_voltages(inputs, control_voltages, self.input_indices, self.control_voltage_indices)
        return self.get_output(y)

    def forward(self, x):
        return self.model(x)

    def get_amplification_value(self):
        return self.info['data_info']['processor']['amplification']

    def _info_consistency_check(self, model_info):
        """ It checks if the model info follows the expected standards.
        If it does not follow the standards, it forces the model to
        follow them and throws an exception. """
        # if type(model_info['activation']) is str:
        #    model_info['activation'] = nn.ReLU()
        if 'D_in' not in model_info['processor']['torch_model_dict']:
            model_info['processor']['torch_model_dict']['D_in'] = 7
            print('WARNING: The model loaded does not define the input dimension as expected. Changed it to default value: 7')
        if 'D_out' not in model_info['processor']['torch_model_dict']:
            model_info['processor']['torch_model_dict']['D_out'] = 1
            print('WARNING: The model loaded does not define the output dimension as expected. Changed it to default value: %d.' % 1)
        if 'hidden_sizes' not in model_info['processor']['torch_model_dict']:
            model_info['processor']['torch_model_dict']['hidden_sizes'] = [90] * 6
            print('WARNING: The model loaded does not define the input dimension as expected. Changed it to default value: %d.' % 90)
        return model_info
# TODO: generalize get_activation function to allow for several options, e.g. relu, tanh, hard-tanh, sigmoid

    def _get_activation(self, activation):
        if type(activation) is str:
            print('Activation function is set as ReLU')
            return nn.ReLU()
        return activation
