""" The accelerator class enables to statically access the accelerator
(CUDA or CPU) that is used in the computer. The aim is to support both platforms seemlessly. """

import torch
from torch import nn
from brainspy.utils.pytorch import TorchUtils


class NeuralNetworkModel(nn.Module):
    """
        The TorchModel class is used to manage together a torch model and its state dictionary. The usage is expected to be as follows
        Loads a neural network from a custom dictionary
        mymodel = TorchModel()
        mymodel.load_model('my_path/my_model.pt')
        mymodel.model
    """
# TODO: Automatically register the data type according to the configurations of the amplification variable of the  info dictionary

    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.load(configs['torch_model_dict'])
        if TorchUtils.get_accelerator_type() == torch.device('cuda'):
            self.raw_model.cuda()

    def load(self, model_info):
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
        self.raw_model = nn.Sequential(*modules)

        print('Model built with the following modules: \n', modules)

    def forward(self, x):
        return self.raw_model(x)

    def _get_activation(self, activation):
        # TODO: generalize get_activation function to allow for several options, e.g. relu, tanh, hard-tanh, sigmoid
        if type(activation) is str:
            print('Activation function is set as ReLU')
            return nn.ReLU()
        return activation
