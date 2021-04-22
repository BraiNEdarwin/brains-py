"""
Module for creating and managing a surrogate model. A surrogate model consists
of a basic neural network, as well as extra effects applied to its output.

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
        Numberof outputs of the network.
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

import torch
import warnings
import collections

import numpy as np
from torch import nn

from brainspy.utils.electrodes import set_effects_from_dict
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
    voltage_ranges : torch.Tensor
        Minimum and maximum voltage for each output.
    output_clipping : Optional[torch.Tensor]
        Minimum and maximum values for clipping the output.
    #TODO: Include also amplification
    """

    def __init__(
        self,
        model_structure: dict,
        model_state_dict: collections.OrderedDict = None,
    ):
        """
        Create a processor, load the model.

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
        default_sim_params: Dictionary containing the keys to the effects that will be simulated.
            #TODO: Complete description
            electrode_no
            activation_electrodes
                    - electrode_no
                    - voltage_ranges
            output_electrodes
                    - electrode_no
                    - amplification
                    - clipping
            Path of the model file.
        model_state_dict: Pytorch's ordered dictionary containing the values for the learnable parameters of the raw model. By default is set to None.
                          If it is not None, the dictionary will be loaded to the raw model.
        """
        super(SurrogateModel, self).__init__()
        self.model = NeuralNetworkModel(model_structure)
        if model_state_dict is not None:
            self.model.load_state_dict(model_state_dict)

    def get_voltage_ranges(self):
        return self.voltage_ranges

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply forward pass on self.model and subsequently apply effects
        if needed.

        Order of effects: amplification - noise - output clipping

        Example
        -------
        >>> smg = SurrogateModel("model.pt")
        >>> smg(torch.tensor([1.0, 2.0, 3.0]))
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
            return torch.clamp(
                x, min=self.output_clipping[0], max=self.output_clipping[1]
            )
        return x

    # For debugging purposes
    def forward_numpy(self, input_matrix: np.array):
        """
        Perform a forward pass of the model without applying effects.
        Works on a numpy tensor: first converted to tensor, then passed
        through the processor, then converted back to numpy.

        Example
        -------
        >>> smg = SurrogateModel("model.pt")
        >>> smg.forward(np.array([1.0, 2.0, 3.0]))
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
        Method to indicate whether this is a hardware processor.

        Returns
        -------
        bool
            False
        """
        return False
