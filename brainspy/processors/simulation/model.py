"""
TODO add module docstring
"""
import warnings
import torch
from torch import nn
from typing import Optional, Dict


class NeuralNetworkModel(nn.Module):
    """
    TODO add class docstring
    """

    # TODO: Automatically register the data type according to the
    # configurations of the amplification variable of the  info dictionary

    def __init__(self, model_info: dict, verbose=False):
        """
        Create a model object.

        Parameters
        ----------
        model_info : dict
            Dictionary containing the model info.
        verbose : bool, optional
            Whether to print certain steps, by default False.
        """
        super(NeuralNetworkModel, self).__init__()
        self.info: Optional[Dict] = None
        self.verbose = verbose
        self.build_model_structure(model_info)

    def build_model_structure(self, model_info: dict):
        """
        Build the model from the info dictionary.
        First perform the consistency check, then set the layers and
        activations.

        Parameters
        ----------
        model_info : dict
            Dictionary containing the weights and activations of the model.
        """
        self.info_consistency_check(model_info)
        hidden_sizes = model_info["hidden_sizes"]
        input_layer = nn.Linear(model_info["D_in"], hidden_sizes[0])
        activ_function = self._get_activation(model_info["activation"])
        output_layer = nn.Linear(hidden_sizes[-1], model_info["D_out"])
        modules = [input_layer, activ_function]

        hidden_layers = zip(hidden_sizes[:-1], hidden_sizes[1:])
        for h_1, h_2 in hidden_layers:
            hidden_layer = nn.Linear(h_1, h_2)
            modules.append(hidden_layer)
            modules.append(activ_function)

        modules.append(output_layer)
        self.raw_model = nn.Sequential(*modules)
        if self.verbose:
            print("Model built with the following modules: \n", modules)

    def set_info_dict(self, info_dict: dict):
        """
        Set the info dictionary of the model.

        Parameters
        ----------
        info_dict : dict
            Info dictionary.
        """
        self.info = info_dict

    def get_info_dict(self) -> Optional[Dict]:
        """
        Get the info dictionary of the model.

        Returns
        -------
        dict
            Info dictionary of the model.

        Raises
        ------
        UserWarning
            If the state dictionary is not set.
        """
        if self.info is None:
            warnings.warn("The info dictionary is empty.")
        return self.info

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Do a forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input data.

        Returns
        -------
        torch.Tensor
            Output data.
        """
        return self.raw_model(x)

    def _get_activation(self, activation):
        """
        Get the activation of the model. If it's a string then return an
        actual activation object.

        Currenly can only read ReLU.

        Parameters
        ----------
        activation : [type]
            [description]

        Returns
        -------
        [type]
            [description]
        """
        # TODO: generalize get_activation function to allow for several
        # options, e.g. relu, tanh, hard-tanh, sigmoid
        if type(activation) is str:
            if self.verbose:
                print("Activation function is set as ReLU")
            return nn.ReLU()
        return activation

    def info_consistency_check(self, model_info: dict):
        """
        Check if the model info follows the expected standards:
        Are activation, input dimension, output dimension, and hidden layer
        number and sizes are defined in the config? If they aren't, set them
        to default values and print a warning.

        Parameters
        ----------
        model_info : dict
            Dictionary of the configs.
        """
        default_in_size = 7
        default_out_size = 1
        default_hidden_size = 90
        default_hidden_number = 6
        default_activation = "relu"
        if not ("activation" in model_info
                and model_info["activation"] in ("relu", "elu")):
            model_info["activation"] = "relu"
            warnings.warn(
                "The model loaded does not define the activation as "
                f"expected. Changed it to default value: {default_activation}."
            )
        if "D_in" not in model_info:
            # Check input dimension.
            model_info["D_in"] = default_in_size
            warnings.warn(
                "The model loaded does not define the input dimension as "
                f"expected. Changed it to default value: {default_in_size}.")
        if "D_out" not in model_info:
            # Check output dimension.
            model_info["D_out"] = default_out_size
            warnings.warn(
                "The model loaded does not define the output dimension as "
                f"expected. Changed it to default value: {default_out_size}.")
        if "hidden_sizes" not in model_info:
            # Check sizes of hidden layers.
            model_info[
                "hidden_sizes"] = default_hidden_size * default_hidden_number
            warnings.warn(
                "The model loaded does not define the hidden layer sizes as "
                f"expected. Changed it to default value: "
                f"{default_hidden_number} layers of {default_hidden_size}.")

