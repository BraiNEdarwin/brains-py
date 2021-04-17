"""
Module for creating and managing a surrogate model. A surrogate model consists
of a basic neural network, as well as extra effects applied to its output.
"""

import warnings

import torch
import numpy as np
from torch import nn

from brainspy.utils.pytorch import TorchUtils
from brainspy.processors.simulation.noise.noise import get_noise


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
    """
    def __init__(self, filename: str):
        """
        Create a processor, load the model.

        Parameters
        ----------
        filename : str
            Path of the model file.
        """
        super(SurrogateModel, self).__init__()
        self.load_base_model(filename)

    # Only used internally.
    def load_base_model(self, filename: str):
        """
        Loads a pytorch model from a directory string.
        Initiate voltage ranges.

        This method is automatically called when creating the SurrogateModel.
        """
        self.model = torch.load(
            filename,
            map_location=TorchUtils.get_device(),
        )
        self._init_voltage_ranges()

    # Only used internally.
    def _init_voltage_ranges(self):
        """
        Load the offset and amplitude from the model and calculate the minimum
        and maximum voltage.

        This method is automatically called when creating the SurrogateModel.
        """
        offset = TorchUtils.format(
            self.model.info["data_info"]["input_data"]["offset"])
        amplitude = TorchUtils.format(
            self.model.info["data_info"]["input_data"]["amplitude"])
        min_voltage = (offset - amplitude).unsqueeze(dim=1)
        max_voltage = (offset + amplitude).unsqueeze(dim=1)
        self.voltage_ranges = torch.cat((min_voltage, max_voltage), dim=1)

    def set_effects(self,
                    amplification=None,
                    output_clipping=None,
                    noise=None,
                    **kwargs):
        """
        Set the amplification, output clipping and noise of the processor.
        Amplification and output clipping are explained in their respective
        methods. Noise is an error which is superimposed on the output of the
        network to give it an element of randomness.

        Order of effects: amplification - noise - output clipping

        Example
        -------
        >>> smg = SurrogateModel("model.pt")
        >>> smg.set_effects(amplification=2.0,
                            output_clipping="default",
                            noise=None)

        Parameters
        ----------
        amplification
            The amplification of the processor. Can be None, a value, or
            'default'. By default None.
        output_clipping
            The output clipping of the processor. Can be None, a value, or
            'default'. By default None.
        noise
            The noise of the processor. Can be None, a string determining
            the type of noise and some args. By default None.
        """
        # Warning, this function used to be called form the init using a
        # configs file. Now it is called externally. To be changed where it
        # corresponds in bspy tasks.
        self.set_amplification(amplification)
        self.set_output_clipping(output_clipping)
        self.noise = get_noise(noise, kwargs)

    def set_amplification(self, value):
        """
        Set the amplification of the processor. The amplificaiton is what the
        output of the neural network is multiplied with after the forward pass.
        Can be None, a value, or 'default', by default None.
        None will not use amplification, a value will set the amplification
        to that value, and the string 'default' will take the data from the
        info dictionary.

        This method is called through the "set_effects" method.

        Parameters
        ----------
        value : None or double or str
            The value of the amplification (None, a value or 'default').
        """
        if value is not None and value == "default":
            self.amplification = TorchUtils.format(
                self.model.info["data_info"]["processor"]["driver"]
                ["amplification"])
        else:
            self.amplification = value

    def set_output_clipping(self, value: torch.Tensor):
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

        Parameters
        ----------
        value : None or double or str
            The value of the output clipping (None, a value or 'default').
        """
        if value is not None and value == "default":
            self.output_clipping = TorchUtils.format(
                self.model.info["data_info"]["clipping_value"])
        else:
            self.output_clipping = value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply forward pass on self.model and subsequently apply effects
        if needed.

        Order of effects: amplification - noise - output clipping

        Example
        -------
        >>> smg = SurrogateModel("model.pt")
        >>> smg.forward(torch.tensor([1.0, 2.0, 3.0]))
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
                               min=self.output_clipping[0],
                               max=self.output_clipping[1])
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

    def get_electrode_no(self):
        """
        Get the number of electrodes of the processor.

        Returns
        -------
        int
            The number of electrodes of the processor.
        """
        return len(self.model.info["data_info"]["input_data"]["offset"])
