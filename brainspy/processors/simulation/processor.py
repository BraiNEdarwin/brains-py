"""
TODO add module docstring
"""

import torch
import warnings

from torch import nn
import numpy as np

from brainspy.utils.pytorch import TorchUtils
from brainspy.processors.simulation.noise.noise import get_noise


class SurrogateModel(nn.Module):
    """
    TODO add class docstring
    """

    # TODO: Automatically register the data type according to the
    # configurations of the amplification variable of the  info dictionary

    def __init__(self, filename: str):
        """
        Create a processor, load the model.

        Parameters
        ----------
        filename : str
            Path of the model file.
        """
        # Configurations are basically the effects, and the path to the file
        # from which the model should be loaded
        super(SurrogateModel, self).__init__()
        self.load_base_model(filename)

    def load_base_model(self, filename: str):
        """
        Loads a pytorch model from a directory string.
        Initiate voltage ranges.
        """
        self.model = torch.load(
            filename,
            map_location=TorchUtils.get_accelerator_type(),
        )
        self._init_voltage_ranges()

    def _init_voltage_ranges(self):
        """
        Load the offset and amplitude from the model and calculate the minimum
        and maximum voltage.
        """
        offset = TorchUtils.get_tensor_from_list(
            self.model["info"]["data_info"]["input_data"]["offset"])
        amplitude = TorchUtils.get_tensor_from_list(
            self.model["info"]["data_info"]["input_data"]["amplitude"])
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

        Parameters
        ----------
        amplification : [type], optional
            [description], by default None
        output_clipping : [type], optional
            [description], by default None
        noise : [type], optional
            [description], by default None
        """
        # Warning, this function used to be called form the init using a
        # configs file. Now it is called externally. To be changed where it
        # corresponds in bspy tasks.
        # noise can be None, a string determining the type of noise and some
        # args.
        self.set_amplification(amplification)
        self.set_output_clipping(output_clipping)
        self.noise = get_noise(noise, kwargs)

    def set_amplification(self, value):
        """
        Set the amplification of the processor. Can be None, a value,
        or 'default'.
        None will not use amplification, a value will set the amplification
        to that value, and the string 'default' will take the data from the
        info dictionary.

        Parameters
        ----------
        value : None or double or str
            The value of the amplification (None, a value or 'default').
        """
        if value is not None and value == "default":
            self.amplification = TorchUtils.get_tensor_from_list(
                self.model.info["data_info"]["processor"]["driver"]
                ["amplification"])
        else:
            self.amplification = value

    def set_output_clipping(self, value):
        """
        Set the output clipping of the processor. Can be None, a value, or
        'default'.
        None will not use clipping, a value will set the clipping to that
        value, and the string 'default' will take the data from the info
        dictionary.

        Parameters
        ----------
        value : None or double or str
            The value of the output clipping (None, a value or 'default').
        """
        if value is not None and value == "default":
            self.output_clipping = TorchUtils.get_tensor_from_list(
                self.model.info["data_info"]["clipping_value"])
        else:
            self.output_clipping = value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply forward pass on self.model and subsequently apply effects
        if needed.

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
                               min=self.clipping_value[0],
                               max=self.clipping_value[1])
        return x

    # For debugging purposes
    def forward_numpy(self, input_matrix: np.array):
        """
        Perform a forward pass of the model without applying effects.
        Works on a numpy tensor: first converted to tensor, then passed
        through the model, then converted back to numpy.

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
            inputs_torch = TorchUtils.get_tensor_from_numpy(input_matrix)
            output = self.forward(inputs_torch)
        return TorchUtils.get_numpy_from_tensor(output)

    def reset(self):
        """
        Reset the processor.
        """
        # TODO write reset function
        warnings.warn(
            "Warning: Reset function in Surrogate Model not implemented.")

    def close(self):
        """
        Close the processor.
        """
        # TODO write close function
        warnings.warn(
            "Warning: Close function in Surrogate Model not implemented.")

    def is_hardware(self):
        """
        Method to indicate whether this is a hardware processor.

        Returns
        -------
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
        if "info" in dir(self.model):
            return len(self.model.info["data_info"]["input_data"]["offset"])
        else:
            warnings.warn(
                "Unable to retrieve electrode number from the info dictionary,"
                "as it has not been loaded yet into the NeuralNetworkModel. "
                "No checks with the info dictionary were performed. To "
                "proceeed safely, make sure that the 'input_electrode_no' "
                "field corresponds to the number of electrodes with which the "
                "NeuralNetworkModel that you are intending to use was trained."
            )
            return None
