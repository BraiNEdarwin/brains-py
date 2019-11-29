""" The accelerator class enables to statically access the accelerator
(CUDA or CPU) that is used in the computer. The aim is to support both platforms seemlessly. """

import torch
import torch.nn as nn
from bspyproc.utils.pytorch import TorchUtils


class TorchModel(nn.Module):
    """
        The TorchModel class is used to manage together a torch model and its state dictionary. The usage is expected to be as follows
        mymodel = TorchModel()
        mymodel.load_model('my_path/my_model.pt')
        mymodel.model
    """
# TODO: Automatically register the data type according to the configurations of the amplification variable of the  info dictionary

    def __init__(self, model_source):
        super().__init__()
        self.info = None
        if type(model_source) is str:
            self.load_model(model_source)
        elif type(model_source) is dict:
            self.build_model(model_source)

        if TorchUtils.get_accelerator_type() == torch.device('cuda'):
            self.model.cuda()

    def load_file(self, data_dir, file_type):
        if file_type == 'pt':
            state_dict = torch.load(data_dir, map_location=TorchUtils.get_accelerator_type())
            info = self._info_consistency_check(state_dict['info'])
            del state_dict['info']
        elif file_type == 'json':
            state_dict = None
            # TODO: Implement loading from a json file
            raise NotImplementedError(f"Loading file from a json file in TorchModel has not been implemented yet. ")
            # info = model_info loaded from a json file
        if 'amplification' not in info.keys():
            info['amplification'] = 1

        return info, state_dict

    def load_model(self, data_dir):
        """Loads a pytorch model from a directory string."""
        self.info, state_dict = self.load_file(data_dir, 'pt')
        self.build_model(self.info)
        self.model.load_state_dict(state_dict)

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
        return TorchUtils.get_numpy_from_tensor(output) * self.info['amplification']

    def forward(self, x):
        return self.model(x)

    def get_amplification_value(self):
        return self.info['amplification']

    def _info_consistency_check(self, model_info):
        """ It checks if the model info follows the expected standards.
        If it does not follow the standards, it forces the model to
        follow them and throws an exception. """
        # if type(model_info['activation']) is str:
        #    model_info['activation'] = nn.ReLU()
        if 'D_in' not in model_info:
            model_info['D_in'] = len(model_info['offset'])
            print('WARNING: The model loaded does not define the input dimension as expected. Changed it to default value: %d.' % len(model_info['offset']))
        if 'D_out' not in model_info:
            model_info['D_out'] = 1
            print('WARNING: The model loaded does not define the output dimension as expected. Changed it to default value: %d.' % 1)
        if 'hidden_sizes' not in model_info:
            model_info['hidden_sizes'] = [90] * 6
            print('WARNING: The model loaded does not define the input dimension as expected. Changed it to default value: %d.' % 90)
        return model_info
# TODO: generalize get_activation function to allow for several options, e.g. relu, tanh, hard-tanh, sigmoid

    def _get_activation(self, activation):
        if type(activation) is str:
            print('Activation function is set as ReLU')
            return nn.ReLU()
        return activation
