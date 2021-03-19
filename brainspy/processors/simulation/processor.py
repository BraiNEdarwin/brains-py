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
    The TorchModel class is used to manage together a torch model and its state dictionary. The usage is expected to be as follows
    mymodel = TorchModel()
    mymodel.load_model('my_path/my_model.pt')
    mymodel.model
    """

    # TODO: Automatically register the data type according to the configurations of the amplification variable of the  info dictionary

    def __init__(
        self, configs
    ):  # Configurations are basically the effects, and the path to the file from which the model should be loaded
        super(SurrogateModel, self).__init__()
        self.model = NeuralNetworkModel()
        self._load(configs)

    def _load(self, configs):
        """Loads a pytorch model from a directory string."""
        self.configs = configs
        # self.info, state_dict = self.load_file(configs["torch_model_dict"])
        self.is_loaded = False
        if "torch_model_dict" in configs:
            state_dict = torch.load(
                configs["torch_model_dict"],
                map_location=TorchUtils.get_accelerator_type(),
            )
            self.model.load_state_dict(state_dict)
            self._init_voltage_ranges()
            self._init_effects()
            self.is_loaded = True
        else:
            warnings.warn(
                "No model dictionary path for the surrogate model has been specified in the configs. A state_dict should be loaded before proceeding with any simulation."
            )

    def _init_voltage_ranges(self):
        offset = TorchUtils.get_tensor_from_list(
            self.model.info["data_info"]["input_data"]["offset"]
        )
        amplitude = TorchUtils.get_tensor_from_list(
            self.model.info["data_info"]["input_data"]["amplitude"]
        )
        min_voltage = (offset - amplitude).unsqueeze(dim=1)
        max_voltage = (offset + amplitude).unsqueeze(dim=1)
        self.voltage_ranges = torch.cat((min_voltage, max_voltage), dim=1)

    def _init_effects(self):
        self.amplification = TorchUtils.get_tensor_from_list(
            self.model.info["data_info"]["processor"]["driver"]["amplification"]
        )
        self.output_clipping = self.configs["output_clipping"]
        self.clipping_value = TorchUtils.get_tensor_from_list(
            self.model.info["data_info"]["clipping_value"]
        )
        self.noise = get_noise(self.configs)

    def forward(self, x):
        x = self.noise(self.model(x) * self.amplification)
        if self.output_clipping:
            return torch.clamp(
                x, min=self.clipping_value[0], max=self.clipping_value[1]
            )
        return x

    def forward_numpy(self, input_matrix):
        with torch.no_grad():
            inputs_torch = TorchUtils.get_tensor_from_numpy(input_matrix)
            output = self.forward(inputs_torch)
        return TorchUtils.get_numpy_from_tensor(output)

    def load_state_dict(self, state_dict, strict=True):
        print("Reached simulation processor")
        super().load_state_dict(state_dict, strict)
        self._init_voltage_ranges()
        self._init_effects()
        self.is_loaded = True

    def reset(self):
        print("Warning: Reset function in Surrogate Model not implemented.")
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

    #     Parameters
    #     ----------
    #     data_dir : str
    #         Directory of the file.

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
    #         data_dir, map_location=TorchUtils.get_accelerator_type()
    #     )
    #     # state_dict is an ordered dictionary.

    #     # Load the info and delete it from the model.
    #     info = state_dict["info"]
    #     del state_dict["info"]
    #     # info is a dictionary; keys are data_info and smg_configs.

    #     # Set amplification to 1 if not specified in file.
    #     if "amplification" not in info["data_info"]["processor"]["driver"]:
    #         info["data_info"]["processor"]["driver"]["amplification"] = 1
    #         warnings.warn(
    #             "The model loaded does not define the amplification; set to 1."
    #         )
    #     if (
    #         "model_structure_dict"
    #         not in info["smg_configs"]["processor"]["torch_model_dict"]
    #     ):
    #         state_dict["model_structure_dict"] = info["smg_configs"]["processor"][
    #             "torch_model_dict"
    #         ]
    #         del info["smg_configs"]["processor"]["torch_model_dict"]
    #     return info, state_dict
