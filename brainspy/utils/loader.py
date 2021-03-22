import warnings

# Only used in processors/simulation/model.py


def info_consistency_check(model_info: dict):
    """
    Check if the model info follows the expected standards:
    Are activation, input dimension, output dimension, and hidden layer number and sizes
    are defined in the config? If they aren't, set them to default values and print a warning.

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
    if not ("activation" in model_info and ["activation"] in ("relu", "elu")):
        model_info["activation"] = "relu"
        warnings.warn(
            "The model loaded does not define the activation as expected. "
            f"Changed it to default value: {default_activation}."
        )
    if "D_in" not in model_info:
        # Check input dimension.
        model_info["D_in"] = default_in_size
        warnings.warn(
            "The model loaded does not define the input dimension as expected. "
            f"Changed it to default value: {default_in_size}."
        )
    if "D_out" not in model_info:
        # Check output dimension.
        model_info["D_out"] = default_out_size
        warnings.warn(
            "The model loaded does not define the output dimension as expected. "
            f"Changed it to default value: {default_out_size}."
        )
    if "hidden_sizes" not in model_info:
        # Check sizes of hidden layers.
        model_info["hidden_sizes"] = (
            default_hidden_size * default_hidden_number
        )
        warnings.warn(
            "The model loaded does not define the hidden layer sizes as expected. "
            f"Changed it to default value: {default_hidden_number} layers of {default_hidden_size}."
        )
