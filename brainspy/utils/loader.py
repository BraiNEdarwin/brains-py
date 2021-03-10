from typing import Tuple, OrderedDict
import warnings
import torch
from brainspy.utils.pytorch import TorchUtils

# Used in processor.py
def load_file(data_dir: str) -> Tuple[dict, OrderedDict]:
    """
    Load a model from a file. Run a consistency check on smg_configs.
    Checks whether the amplification of the processor is set in the config; if not, set it to 1.

    Example
    -------
    >>> load_file("model.pt")
    (info, state_dict)

    In this case 'info' contains information about the model and 'state_dict' contains the weights
    of the network, referring to the model in "model.pt".

    Parameters
    ----------
    data_dir : str
        Directory of the file.

    Returns
    -------
    info : dict
        Dictionary containing the settings.
    state_dict : dict
        State dictionary of the model, containing the weights and biases
        of the network.
    """
    # Load model; contains weights (+biases) and info.
    state_dict = torch.load(data_dir, map_location=TorchUtils.get_accelerator_type())
    # state_dict is an ordered dictionary.

    # Load the info and delete it from the model.
    info = state_dict["info"]
    del state_dict["info"]
    # info is a dictionary; keys are data_info and smg_configs.

    # Run consistency check (see docstring of that method).
    info["smg_configs"] = info_consistency_check(info["smg_configs"])

    # Set amplification to 1 if not specified in file.
    if "amplification" not in info["data_info"]["processor"]:
        info["data_info"]["processor"]["amplification"] = 1
        warnings.warn("The model loaded does not define the amplification; set to 1.")
    return info, state_dict


# Only used in this file.
def info_consistency_check(model_info: OrderedDict) -> OrderedDict:
    """
    Check if the model info follows the expected standards:
    Are input dimension, output dimension, and hidden layer number and sizes
    are defined in the config? If they aren't, set them to default values and
    print a warning.
    Used on the smg_configs specifically.

    Parameters
    ----------
    model_info : dict
        Dictionary of the configs.

    Returns
    -------
    model_info : dict
        A possibly altered version of the input dictionary.
    """
    default_in_size = 7
    default_out_size = 1
    default_hidden_size = 90
    default_hidden_number = 6
    if "D_in" not in model_info["processor"]["torch_model_dict"]:
        # Check input dimension.
        model_info["processor"]["torch_model_dict"]["D_in"] = default_in_size
        warnings.warn(
            "The model loaded does not define the input dimension as expected. "
            f"Changed it to default value: {default_in_size}."
        )
    if "D_out" not in model_info["processor"]["torch_model_dict"]:
        # Check output dimension.
        model_info["processor"]["torch_model_dict"]["D_out"] = default_out_size
        warnings.warn(
            "The model loaded does not define the output dimension as expected. "
            f"Changed it to default value: {default_out_size}."
        )
    if "hidden_sizes" not in model_info["processor"]["torch_model_dict"]:
        # Check sizes of hidden layers.
        model_info["processor"]["torch_model_dict"]["hidden_sizes"] = (
            default_hidden_size * default_hidden_number
        )
        warnings.warn(
            "The model loaded does not define the hidden layer sizes as expected. "
            f"Changed it to default value: {default_hidden_number} layers of {default_hidden_size}."
        )
    return model_info
