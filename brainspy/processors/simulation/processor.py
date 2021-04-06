""" The accelerator class enables to statically access the accelerator
(CUDA or CPU) that is used in the computer. The aim is to support both platforms seemlessly. """

import torch
import warnings

from torch import nn
from typing import Tuple, OrderedDict

from brainspy.utils.pytorch import TorchUtils
from brainspy.processors.simulation.noise.noise import get_noise
from brainspy.processors.simulation.model import NeuralNetworkModel


class SurrogateModel(nn.Module):
    """
    Remove these old comments below:
    The TorchModel class is used to manage together a torch model and its state dictionary. The usage is expected to be as follows
    mymodel = TorchModel()
    mymodel.load_model('my_path/my_model.pt')
    mymodel.model
    """

    # TODO: Automatically register the data type according to the configurations of the amplification variable of the  info dictionary

    def __init__(
        self, filename
    ):  # Configurations are basically the effects, and the path to the file from which the model should be loaded
        super(SurrogateModel, self).__init__()
        self.load_base_model(filename)

    def load_base_model(self, filename):
        """Loads a pytorch model from a directory string."""
        self.model = torch.load(
            filename,
            map_location=TorchUtils.get_device(),
        )
        self._init_voltage_ranges()

    def _init_voltage_ranges(self):
        offset = TorchUtils.format(self.model.info["data_info"]["input_data"]["offset"])
        amplitude = TorchUtils.format(
            self.model.info["data_info"]["input_data"]["amplitude"]
        )
        min_voltage = (offset - amplitude).unsqueeze(dim=1)
        max_voltage = (offset + amplitude).unsqueeze(dim=1)
        self.voltage_ranges = torch.cat((min_voltage, max_voltage), dim=1)

    def set_effects(
        self, amplification=None, output_clipping=None, noise=None, **kwargs
    ):
        # Warning, this function used to be called form the init using a configs file. Now it is called externally. To be changed where it corresponds in bspy tasks.
        # Amplification can be None, a value, or 'default'. None will not use amplification, a value will set the amplification to that value, and the string 'default' will take the data from the info dictionary.
        # clipping can be None, a value, or 'default'. None will not use clipping, a value will set the clipping to that value, and the string 'default' will take the data from the info dictionary.
        # noise can be None, a string determining the type of noise and some args.
        self.set_amplification(amplification)
        self.set_output_clipping(output_clipping)
        self.noise = get_noise(noise, kwargs)

    def set_amplification(self, value):
        if value is not None and value == "default":
            self.amplification = TorchUtils.format(
                self.model.info["data_info"]["processor"]["driver"]["amplification"]
            )
        else:
            self.amplification = value

    def set_output_clipping(self, value):
        if value is not None and value == "default":
            self.output_clipping = TorchUtils.format(
                self.model.info["data_info"]["clipping_value"]
            )
        else:
            self.output_clipping = value

    def forward(self, x):
        x = self.model(x)
        if self.amplification is not None:
            x = x * self.amplification
        if self.noise is not None:
            x = self.noise(x)
        if self.output_clipping is not None:
            return torch.clamp(
                x, min=self.clipping_value[0], max=self.clipping_value[1]
            )
        return x

    # For debugging purposes
    def forward_numpy(self, input_matrix):
        with torch.no_grad():
            inputs_torch = TorchUtils.format(input_matrix)
            output = self.forward(inputs_torch)
        return TorchUtils.to_numpy(output)

    def reset(self):
        warnings.warn("Warning: Reset function in Surrogate Model not implemented.")
        pass

    def close(self):
        # print('The surrogate model does not have a closing function. ')
        pass

    def is_hardware(self):
        return False

    def get_electrode_no(self):
        if "info" in dir(self.model):
            return len(self.model.info["data_info"]["input_data"]["offset"])
        else:
            warnings.warn(
                "Unable to retrieve electrode number from the info dictionary, as it has not been loaded yet into the NeuralNetworkModel. No checks with the info dictionary were performed. To proceeed safely, make sure that the 'input_electrode_no' field corresponds to the number of electrodes with which the NeuralNetworkModel that you are intending to use was trained."
            )
            return None

    # def load_file(self, data_dir: str) -> Tuple[dict, OrderedDict]:
    #     """
    #     Load a model from a file. Run a consistency check on smg_configs.
    #     Checks whether the amplification of the processor is set in the config; if not, set it to 1.

    #     Example
    #     -------
    #     >>> load_file("model.pt")
    #     (info, state_dict)

    #     In this case 'info' contains information about the model and 'state_dict' contains the weights
    #     of the network, referring to the model in "model.pt".

    #    Returns
    #    -------
    #    info : dict
    #        Dictionary containing the settings.
    #    state_dict : dict
    #        State dictionary of the model, containing the weights and biases
    #        of the network.
    #    """
    #    # Load model; contains weights (+biases) and info.
    #    state_dict = torch.load(
    #        data_dir, map_location=TorchUtils.get_device()
    #    )
    #    # state_dict is an ordered dictionary.

    #     Returns
    #     -------
    #     info : dict
    #         Dictionary containing the settings.
    #     state_dict : dict
    #         State dictionary of the model, containing the weights and biases
    #         of the network.
    #     """
    #     # Load model; contains weights (+biases) and info.
    #     state_dict = torch.load(
    #         data_dir, map_location=TorchUtils.get_device()
    #     )
    #     # state_dict is an ordered dictionary.

    #    # Set amplification to 1 if not specified in file.
    #    if "amplification" not in info["data_info"]["processor"]:
    #        info["data_info"]["processor"]["amplification"] = 1
    #        warnings.warn(
    #            "The model loaded does not define the amplification; set to 1."
    #        )
    #    return info, state_dict
