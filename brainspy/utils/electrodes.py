import warnings

# from brainspy.processors.hardware.processor import HardwareProcessor
from brainspy.processors.simulation.noise.noise import get_noise
from brainspy.utils.pytorch import TorchUtils


# TODO: Add description of this method
def set_effects_from_dict(processor, info, configs):
    return set_effects(
        processor,
        info,
        get_key(configs, "voltage_ranges"),
        get_key(configs, "amplification"),
        get_key(configs, "output_clipping"),
        get_key(configs, "noise"),
    )


# TODO: Add description of this method
def get_key(configs, effect_key):
    if effect_key in configs:
        return configs[effect_key]
    if effect_key != "noise":
        return "default"
    return None


def set_effects(
    processor,
    info,
    voltage_ranges="default",
    amplification="default",
    output_clipping="default",
    noise_configs=None,
):
    """
    Set the amplification, output clipping and noise of the processor.
    Amplification and output clipping are explained in their respective
    methods. Noise is an error which is superimposed on the output of the
    network to give it an element of randomness.

    Order of effects: amplification - noise - output clipping

    Example
    -------
    # TODO: update example of loading model
    >>> smg = SurrogateModel("model.pt")
    >>> smg.set_effects(amplification=2.0,
                        output_clipping="default",
                        noise=None)

    Parameters
    ----------
    voltage_ranges:
        Voltage ranges of the activation electrodes. Can be a value or 'default'.
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

    processor.amplificaiton = set_amplification(info, amplification)
    processor.output_clipping = set_output_clipping(info, output_clipping)

    if processor.is_hardware():
        warnings.warn(
            f"The hardware setup has been initialised with regard to a model trained with the following parameters. Please make sure that the configurations of your hardware setup match these values: \n\t * An amplification correction of {processor.amplification}\n\t * a clipping value range between {processor.clipping_value}\n\t * and voltage ranges within {processor.voltage_ranges.T} "
        )
    else:
        processor.voltage_ranges = set_voltage_ranges(info, voltage_ranges)
        processor.noise = get_noise(noise_configs)
    return processor


def set_voltage_ranges(info, value):
    # TODO: Document this function.
    if value is not None and value == "default":
        return TorchUtils.format(info["activation_electrodes"]["voltage_ranges"])
    elif value is not None:
        # TODO: Add warning to let the user know that the voltage ranges have been changed.
        assert value.shape == info["activation_electrodes"]["voltage_ranges"].shape
        return TorchUtils.format([value])
    return value


def set_amplification(info, value):
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
        return TorchUtils.format([info["output_electrodes"]["amplification"]])
    elif value is not None:
        # TODO: Add warning to let the user know that the original amplification has been changed.
        return TorchUtils.format([value])
    return value


def set_output_clipping(info, value):
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
        return TorchUtils.format([info["output_electrodes"]["clipping_value"]])
    elif value is not None:
        # TODO: Add warning to let the user know that the output clipping ranges have been changed.
        return TorchUtils.format([value])
    return value
