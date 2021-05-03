"""
Module for creating and managing a surrogate model. A surrogate model consists
of a basic neural network, as well as extra effects applied to its output:
amplification, output clipping, noise.

Note: this should be moved
smg_configs
-----------
results_base_dir : str
    Path for storing results.
hyperparameters:
    epochs : int
        Number of epochs.
    learning_rate : float
        Learning rate.
model_architecture:
    hidden_sizes : list[int]
        Number of neurons in each hidden layer of the network.
    D_in : int
        Number of inputs of the network.
    D_out : int
        Number of outputs of the network.
    activation : str
        Type of activation used in the network, for example "relu".
data:
    postprocessed_data_path : str
        Path for storing processed data.
    steps : int
        Step size for sampling data.
    batch_size : int
        Number of data points per batch.
    worker_no : int
        Distribute data loading to subprocesses. Value 0 will only use the
        main process, while value >0 will create that number of subprocesses
        and not use the main process.
        https://pytorch.org/docs/stable/data.html
    pin_memory : bool
        Use CUDA pinned memory when loading data.
        https://pytorch.org/docs/stable/data.html
    split_percentages : list[float]
        Percentage of data in each category [train, validation, test].
"""

import warnings
import collections
from typing import Optional, Union

import torch
import numpy as np
from torch import nn

from brainspy.utils.pytorch import TorchUtils
from brainspy.processors.simulation.noise.noise import get_noise
from brainspy.processors.simulation.model import NeuralNetworkModel


class SurrogateModel(nn.Module):
    """
    Consists of nn model with added effects: amplification, output clipping,
    and noise. The different effects are explained in their respective methods.

    Attributes
    ----------
    model : torch.nn.Module
        The neural network the surrogate model works on.
    voltage_ranges : Optional[torch.Tensor]
        Minimum and maximum voltage for each input.
    output_clipping : Optional[torch.Tensor]
        Minimum and maximum values for clipping the output.
    noise : Optional[Noise]
        Noise object that is applied to the output of the network (for example
        Gaussian noise).
    amplification : Optional[torch.Tensor]
        Amplification applied to the output of the network.
    """
    def __init__(
        self,
        model_structure: dict,
        model_state_dict: collections.OrderedDict = None,
    ):
        """
        Create a processor, load the model.

        Example
        -------
        >>> model_structure = {
                "D_in": 7,
                "D_out": 1,
                "activation": "relu",
                "hidden_sizes": [20, 20, 20]
            }
        >>> SurrogateModel(model_structure)

        In this example a SurrogateModel is instantiated with 7 input
        electrodes, 1 output electrode, ReLU activation and 3 hidden layers
        of 20 neurons each. Since no state dictionary is given, the weights
        of the network will be random.

        Parameters
        ----------
        model_structure : Dictionary containing the model structure.
            D_in : int
                Number of inputs (electrodes).
            D_out : int
                Number of outputs (electrodes).
            activation : str
                Type of activation. Supported activations are "relu", "elu",
                "tanh", "hard-tanh", or "sigmoid".
            hidden_sizes : list[int]
                Sizes of the hidden layers.
        model_state_dict : collections.OrderedDict
            Pytorch's ordered dictionary containing the values for the
            learnable parameters of the raw model. By default is set to None.
            If it is None, the network will be initialized with random
            weights. If it is not None, the dictionary will be loaded to the
            raw model.
        """
        super(SurrogateModel, self).__init__()
        self.model = NeuralNetworkModel(model_structure)
        if model_state_dict is not None:
            self.model.load_state_dict(model_state_dict)

    def get_voltage_ranges(self) -> Optional[torch.Tensor]:
        """
        Return the voltage ranges of the processor.
        Will return None if not set yet.

        Returns
        -------
        torch.Tensor
            The voltage ranges of the processor.
        """
        return self.voltage_ranges

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply forward pass on self.model and subsequently apply effects
        if needed.

        Order of effects: amplification - noise - output clipping

        Example
        -------
        >>> sm.forward(torch.tensor([1.0, 2.0, 3.0]))
        torch.Tensor([4.0])

        Parameters
        ----------
        x : torch.Tensor
            Input data.

        Returns
        -------
        x : torch.tensor
            Output data.
        """
        x = self.model(x)
        if self.amplification is not None:
            x = x * self.amplification
        if self.noise is not None:
            x = self.noise(x)
        if self.output_clipping is not None:
            return torch.clamp(x,
                               min=self.output_clipping[1],
                               max=self.output_clipping[0])
        return x

    # For debugging purposes
    def forward_numpy(self, input_matrix: np.array) -> np.array:
        """
        Perform a forward pass of the model without applying effects and
        without calculating the gradient.
        Works on a numpy tensor: first converted to tensor, then passed
        through the processor, then converted back to numpy.

        Example
        -------
        >>> sm.forward(np.array([1.0, 2.0, 3.0]))
        np.array([4.0])

        Parameters
        ----------
        input_matrix : np.array
            Input data.

        Returns
        -------
        np.array
            Data after forward pass.
        """
        with torch.no_grad():
            inputs_torch = TorchUtils.format(input_matrix)
            output = self.forward(inputs_torch)
        return TorchUtils.to_numpy(output)

    def reset(self):
        """
        Reset the processor. Since this is a simulation model, this does
        nothing.
        """
        warnings.warn("Simulation processor does not reset.")

    def close(self):
        """
        Close the processor. Since this is a simulation model, this does
        nothing.
        """
        warnings.warn("Simulation processor does not close.")

    def is_hardware(self):
        """
        Method to indicate whether this is a hardware processor. Returns
        False.

        Returns
        -------
        bool
            False
        """
        return False

    def set_effects_from_dict(self, info: dict, configs: dict):
        """
        Set the effects of the processor from a dictionary (voltage_ranges,
        amplification, output_clipping, noise). See set_effects for more
        details.
        Need to provide info dictionary in case configs are set to "default".

        Effect values are provided as lists and stored as tensors.

        Example
        -------
        >>> configs = {"amplification": [2.0],
                       "voltage_ranges": [[1.0, 2.0]] * 7,
                       "output_clipping": [2.0, 1.0]}
        >>> sm.set_effects_from_dict(info_dict, configs)

        Parameters
        ----------
        info : dict
            Info dictionary of the processor.
            electrode_no : int
                Total number of electrodes.
            activation_electrodes:
                electrode_no : int
                    Number of activation electrodes.
                voltage_ranges : list[list[float]]
                    Voltage ranges for the input (activation) electrodes.
                    Should contain a pair of values (min and max) for each
                    input.
            output_electrodes:
                electrode_no : int
                    Number of output electrodes.
                amplification : list[float]
                    Amplification applied to the output electrodes.
                output_clipping : list[float]
                    Clipping applied to the output electrodes (2 elements:
                    maximum and minimum value in that order).
        configs : dict
            Dictionary containing the desired effects.
            amplification : list[float]
                Optional, ampfliciation to be applied to the output of the
                network.
            voltage_ranges : list[list[float]]
                Optional, voltage ranges of the input electrodes.
            output_clipping : list[float]
                Clipping applied to the output electrodes.
            noise : dict
                Optional, noise to be applied to the output of the network.
        """
        return self.set_effects(
            info,
            self.get_key(configs, "voltage_ranges"),
            self.get_key(configs, "amplification"),
            self.get_key(configs, "output_clipping"),
            self.get_key(configs, "noise"),
        )

    def get_key(self, configs: dict,
                effect_key: str) -> Optional[Union[str, float]]:
        """
        Get a key from a dictionary, if the dictionary does not contain the
        key, return 'default' (or None if the key is 'noise').

        Example
        -------
        >>> configs = {"amplification": [2.0]}
        >>> sm.get_key("amplification")
        [2.0]
        >>> sm.get_key("output_clipping")
        "default"
        >>> sm.get_key("noise")
        None

        Parameters
        ----------
        configs : dict
            Dictionary from which a value is needed.
        effect_key : str
            The key for which the value is needed.

        Returns
        -------
        str or list[float] or None
            The value of the key or 'default' or None.
        """
        if effect_key in configs:
            return configs[effect_key]
        if effect_key != "noise":
            return "default"
        return None

    def set_effects(
        self,
        info: dict,
        voltage_ranges="default",
        amplification="default",
        output_clipping="default",
        noise_configs=None,
    ):
        """
        Set the amplification, output clipping and noise of the processor.
        Amplification and output clipping are explained in their respective
        methods. Noise is an error which is superimposed on the output of the
        network to give it an element of randomness. See noise.py for more
        information.

        If any of the inputs for the effects are 'default' the value will be
        taken from the info dictionary.

        Effect values are provided as lists and stored as tensors.

        Order of effects: amplification - noise - output clipping

        Example
        -------
        >>> sm.set_effects(info,
                           voltage_ranges="default",
                           amplification=[2.0]),
                           output_clipping="default",
                           noise={"noise_type": "gaussian", "variance": 1.0})

        Parameters
        ----------
        info : dict
            Dictionary with the info of the model. Documented in
            set_effects_from_dict.
        voltage_ranges : str or list[list[float]]
            Voltage ranges of the activation electrodes. Can be a value or
            'default'. By default 'default'.
        amplification : str or list[float]
            The amplification of the processor. Can be None, a value, or
            'default'. By default 'default'.
        output_clipping : str or list[float]
            The output clipping of the processor. Can be None, a value, or
            'default'. By default 'default'.
        noise_configs : dict
            The noise of the processor. Can be None (will generate no noise)
            or a dictionary with keys "noise_type" and "variance" (the latter
            only in case of Gaussian noise).
        """
        self.set_amplification(info, amplification)
        self.set_output_clipping(info, output_clipping)
        self.set_voltage_ranges(info, voltage_ranges)
        self.noise = get_noise(noise_configs)

    def set_voltage_ranges(self, info: dict, value):
        """
        Set the voltage ranges of the processor to a given value or get the
        value from the info dictionary (if value is 'default'). If value is
        None, nothing happens.

        This method is called through the set_effects method.

        Example
        -------
        >>> sm.set_voltage_ranges(info, [[1.0, 2.0]] * 7)

        Here the voltage range is set to 1.0 to 2.0 for each of the 7
        activation electrodes.

        Parameters
        ----------
        info : dict
            Dictionary with information of the processor. Documented in
            set_effects_from_dict.
        value : str or list[list[float]]
            Desired value for the voltage ranges, can also be None (nothing
            happens) or 'default' (get the value from the info dict).

        Raises
        ------
        AssertionError
            If the list given has the wrong length.
        UserWarning
            If the voltage ranges are changed.
        """
        if value is not None and value == "default":
            self.voltage_ranges = TorchUtils.format(
                info["activation_electrodes"]["voltage_ranges"])
        elif value is not None:
            assert len(value) == info["activation_electrodes"]["electrode_no"]
            warnings.warn(
                "Voltage ranges of surrogate model have been changed.")
            self.voltage_ranges = TorchUtils.format(value)

    def set_amplification(self, info: dict, value):
        """
        Set the amplification of the processor. The amplification is what the
        output of the neural network is multiplied with after the forward pass.
        Can be None, a value, or 'default'.
        None will not use amplification, a value will set the amplification
        to that value, and the string 'default' will take the data from the
        info dictionary.

        This method is called through the "set_effects" method.

        Example
        -------
        >>> sm.set_amplification(info, [2.0])

        Parameters
        ----------
        info : dict
            Dictionary with information of the processor. Documented in
            set_effects_from_dict.
        value : None or float or str
            The value of the amplification (None, a value or 'default').

        Raises
        ------
        AssertionError
            If the list given has the wrong length.
        UserWarning
            If the amplification is changed.
        """
        if value is not None and value == "default":
            self.amplification = TorchUtils.format(
                info["output_electrodes"]["amplification"])
        elif value is not None:
            assert len(value) == info["output_electrodes"]["electrode_no"]
            warnings.warn("Amplification of surrogate model has been changed.")
            self.amplification = TorchUtils.format(value)

    def set_output_clipping(self, info: dict, value):
        """
        Set the output clipping of the processor. Output clipping means to
        clip the output to a certain range. Any output above that range will
        be replaced with the maximum and any output below will be set to the
        minimum.
        Can be None, a value, or 'default'.
        None will not use clipping, a value will set the clipping to that
        value, and the string 'default' will take the data from the info
        dictionary.

        This method is called through the "set_effects" method.

        Example
        -------
        >>> sm.set_output_clipping(info, [2.0, 1.0])

        Parameters
        ----------
        info : dict
            Dictionary with information of the processor. Documented in
            set_effects_from_dict.
        value : None or np.array or str
            The value of the output clipping (None, a value or 'default').

        Raises
        ------
        AssertionError
            If the list given has the wrong length.
        UserWarning
            If the output clipping values are changed.
        """
        if value is not None and value == "default":
            self.output_clipping = TorchUtils.format(
                info["output_electrodes"]["output_clipping"])
        elif value is not None:
            assert len(value) == 2
            warnings.warn(
                "Output clipping values of surrogate model have been changed.")
            self.output_clipping = TorchUtils.format(value)
