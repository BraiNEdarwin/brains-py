"""
Placeholder module docstring.
"""

import warnings
from typing import OrderedDict

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
