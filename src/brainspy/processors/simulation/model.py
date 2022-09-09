"""
Module for creating and using a neural network model.
"""
import warnings

import torch
from torch import nn


class NeuralNetworkModel(nn.Module):
    """
    A class for predicting the raw input/output relationship of a DNPU hardware device
    with a neural network model. It consists of a custom length fully connected layer.

    Attributes:
    model_structure : dict
        Dictionary containing the model structure; keys explained in
        constructor method.
    raw_model : nn.Sequential
        Torch object containing the layers and activations of the network.
    """
    def __init__(self, model_structure: dict):
        """
        Create a model object.
        Parameters
        ----------
        model_structure : dict
            Dictionary containing the model structure.
            D_in : int
                Number of inputs (electrodes).
            D_out : int
                Number of outputs (electrodes).
            activation : str
                Type of activation. Supported activations are "relu", "elu",
                "tanh", "hard-tanh", or "sigmoid".
            hidden_sizes : list[int]
                Sizes of the hidden layers.
        """
        super(NeuralNetworkModel, self).__init__()
        self.build_model_structure(model_structure)

    def build_model_structure(self, model_structure: dict):
        """
        Build the model from the structure dictionary.
        First perform the consistency check, then set the layers and
        activations.
        This method is called when an object is created.
        Parameters
        ----------
        model_structure : dict
            Dictionary containing the weights and activations of the model.
            The following keys are required :
                D_in : int
                    Number of inputs (electrodes).
                D_out : int
                    Number of outputs (electrodes).
                activation : str
                    Type of activation. Supported activations are "relu", "elu",
                    "tanh", "hard-tanh", or "sigmoid".
                hidden_sizes : list[int]
                    Sizes of the hidden layers.
        """
        if model_structure is None:
            model_structure = {}
        self.structure_consistency_check(model_structure)
        hidden_sizes = model_structure["hidden_sizes"]
        input_layer = nn.Linear(model_structure["D_in"], hidden_sizes[0])
        activ_function = self._get_activation(model_structure["activation"])
        output_layer = nn.Linear(hidden_sizes[-1], model_structure["D_out"])
        modules = [input_layer, activ_function]

        hidden_layers = zip(hidden_sizes[:-1], hidden_sizes[1:])
        for h_1, h_2 in hidden_layers:
            hidden_layer = nn.Linear(h_1, h_2)
            modules.append(hidden_layer)
            modules.append(activ_function)

        modules.append(output_layer)
        self.raw_model = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Do a forward pass through the raw neural network model simulating the
        input-output relationship of a device.

        Example
        -------
        >>> model = NeuralNetworkModel(d)
        >>> model.forward(torch.tensor([1.0, 2.0, 3.0]))
        torch.Tensor([4.0])
        Parameters
        ----------
        x : torch.Tensor
            Input data.
        Returns
        -------
        torch.Tensor
            Output data.
        """
        assert type(x) is torch.Tensor, "Input to the forward pass can only be a Pytorch tensor"
        return self.raw_model(x)

    def _get_activation(self, activation: str):
        """
        Get the activation of the model. If it's a string then return an
        actual activation object, if that type of activation is implemented.
        If string is not recognized, raise warning and return relu.
        Example
        -------
        >>> model = NeuralNetworkModel(d)
        >>> model._get_activation("tanh")
        nn.Tanh
        Parameters
        ----------
        activation : str
            Type of activation, can be relu, elu, tanh, hard-tanh, sigmoid.
        Returns
        -------
        activation
            An activation object.
        Raises
        ------
        UserWarning
            If activation string is not recognized.
        """
        if activation == "relu":
            return nn.ReLU()
        elif activation == "elu":
            return nn.ELU()
        elif activation == "tanh":
            return nn.Tanh()
        elif activation == "hard-tanh":
            return nn.Hardtanh()
        elif activation == "sigmoid":
            return nn.Sigmoid()
        else:
            warnings.warn("Activation not recognized, applying ReLU")
            return nn.ReLU()

    def structure_consistency_check(self, model_structure: dict):
        """
        Check if the model structure follows the expected standards:
        Are activation, input dimension, output dimension, and hidden layer
        number and sizes are defined in the config? If they aren't, set them
        to default values and print a warning.
        This method is called when an object is created.
        Parameters
        ----------
        model_structure : dict
            Dictionary of the configs.
        Raises
        ------
        UserWarning
            If a parameter is not in the expected format.
        """
        default_in_size = 7
        default_out_size = 1
        default_hidden_size = 90
        default_hidden_number = 6
        default_activation = "relu"
        if not ("activation" in model_structure):
            model_structure["activation"] = "relu"
            warnings.warn(
                "The model loaded does not define the activation as "
                f"expected. Changed it to default value: {default_activation}."
            )
        if "D_in" not in model_structure:
            # Check input dimension.
            model_structure["D_in"] = default_in_size
            warnings.warn(
                "The model loaded does not define the input dimension as "
                f"expected. Changed it to default value: {default_in_size}.")
        else:
            D_in = model_structure.get('D_in')
            assert (type(D_in) == int)
            assert (D_in > 0), "D_in cannot be negative nor zero"

        if "D_out" not in model_structure:
            # Check output dimension.
            model_structure["D_out"] = default_out_size
            warnings.warn(
                "The model loaded does not define the output dimension as "
                f"expected. Changed it to default value: {default_out_size}.")
        else:
            D_out = model_structure.get('D_out')
            assert (type(D_out) == int)
            assert (D_out > 0), "D_out cannot be negative nor zero"

        if "hidden_sizes" not in model_structure:
            # Check sizes of hidden layers.
            model_structure["hidden_sizes"] = [default_hidden_size
                                               ] * default_hidden_number
            warnings.warn(
                "The model loaded does not define the hidden layer sizes as "
                f"expected. Changed it to default value: "
                f"{default_hidden_number} layers of {default_hidden_size}.")
        else:
            hidden_sizes = model_structure.get('hidden_sizes')
            assert (type(hidden_sizes) == list)
            for i in hidden_sizes:
                assert (type(i) == int), "Values for hidden sizes should be int"
                assert i > 0, "Values lower than 1 not allowed for hidden sizes"
