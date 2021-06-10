"""
Class to retrieve data specified in a dictionary used to train a  model.
The class methods require a configuration dictionary with data keys specific to do a task.
It can be used to get an optimizer,fitness function,driver or an algorithm for training a model.
"""

import torch
import collections
from brainspy.processors.hardware.drivers.nidaq import CDAQtoNiDAQ
from brainspy.processors.hardware.drivers.cdaq import CDAQtoCDAQ
import brainspy.algorithms.modules.signal as criterion
import brainspy.algorithms.modules.optim as bspyoptim
from brainspy.algorithms.ga import train as train_ga
from brainspy.algorithms.gd import train as train_gd


def get_criterion(configs: dict):
    """
    Gets the fitness function used in GA from the module FitnessFunctions
    The fitness functions must take two arguments, the outputs of the black-box and the target
    and must return a numpy array of scores of size len(outputs).

    Parameters
    ----------
    configs (dict):  configurations for the fitness function

    Returns
    -------
    fitness fucntion (method/function): A function that is defined in the signal class.
                                        These are a set of functions to measure separability and similarity of signals

    Example
    --------

    configs = {"criterion" : "corr_fit" }
    criterion = get_criterion(configs)

    """
    if configs["criterion"] == "corr_fit":
        return criterion.corr_fit
    elif configs["criterion"] == "accuracy_fit":
        return criterion.accuracy_fit
    elif configs["criterion"] == "corrsig_fit":
        return criterion.corrsig_fit
    elif configs["criterion"] == "fisher":
        return criterion.fisher
    elif configs["criterion"] == "fisher_fit":
        return criterion.fisher_fit
    elif configs["criterion"] == "corrsig":
        return criterion.corrsig
    elif configs["criterion"] == "sqrt_corrsig":
        return criterion.sqrt_corrsig
    elif configs["criterion"] == "fisher_added_corr":
        return criterion.fisher_added_corr
    elif configs["criterion"] == "fisher_multipled_corr":
        return criterion.fisher_multipled_corr
    elif configs["criterion"] == "bce":
        return torch.nn.BCEWithLogitsLoss()
    elif configs["criterion"] == "sigmoid_nn_distance":
        return criterion.sigmoid_nn_distance
    else:
        raise NotImplementedError(
            f"Criterion {configs['criterion']} is not recognised."
        )


def get_optimizer(model: object, configs: dict):
    """
    Gets an optimizer object which include added information to train a specific model.

    Parameters
    ----------
    model (nn.Module object): An nn.Module object which can be a DNPU,Processor or a Surrogate Model object
    configs (dict): configurations for the model

    Returns
    -------
    class object: Returns and optimizer object which can be a GeneticOptimizer or an Adam optimizer

    Example
    --------
    configs = {"optimizer" : "genetic"}
    model = CustomModel()
    optimizer = get_optimizer(model,configs)

    """
    if configs["optimizer"] == "genetic":
        # TODO: get gene ranges from model
        if "gene_range" in configs:
            return bspyoptim.GeneticOptimizer(
                configs["gene_range"], configs["partition"], configs["epochs"]
            )
        else:
            # Only a single device is supported, therefore model.get_control_ranges()[0]
            return bspyoptim.GeneticOptimizer(
                model.get_control_ranges()[0], configs["partition"], configs["epochs"]
            )
    elif configs["optimizer"] == "elm":
        print("ELM optimizer not implemented yet")
        # return get_adam(parameters, configs)
    elif configs["optimizer"] == "adam":
        return get_adam(model, configs)
    else:
        assert False, "Optimiser name {configs['optimizer']} not recognised. Please try"


def get_adam(model: object, configs: dict):
    """
    To get an Adam optimizer object which include added information to train a specific model.
    It is for first-order gradient-based optimization of stochastic objective functions, based on adaptive estimates of lower-order moments.

    Parameters
    ----------
    model (nn.Module object): An nn.Module object which can be a DNPU,Processor or a SurrogateModel object
    configs (dict): configurations of the model

    Returns
    -------
    class object: Returns and optimizer Adam optimizer object

    Example
    --------
    configs = {"optimizer" : "adam"}
    model = CustomModel()
    optimizer = get_adam(model,configs)

    """
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    print("Prediction using ADAM optimizer")
    if "betas" in configs.keys():
        print("Set betas to values from the config file: ")
        print(*configs["betas"], sep=", ")
        return torch.optim.Adam(
            parameters, lr=configs["learning_rate"], betas=configs["betas"]
        )

    else:
        return torch.optim.Adam(parameters, lr=configs["learning_rate"])


def get_algorithm(configs: dict):
    """
    To get a train function, either GA - genetic algorithm or GD - Gradient Descent, based on the configurations dictionary.
    Genetic Algorithm : In computer science and operations research, a genetic algorithm (GA) is a meta-heuristic inspired by the process of natural selection that belongs to the larger class of evolutionary algorithms (EA).
                        Genetic algorithms are commonly used to generate high-quality solutions to optimization and search problems by relying on bio-inspired operators such as mutation, crossover and selection.
                        This algorithm is suitable for experiments with reservoir computing.

    Gradient Descent : Gradient descent is a first-order iterative optimization algorithm for finding the minimum of a function.
                       To find a local minimum of a function using gradient descent, one takes steps proportional to the negative of the gradient (or approximate gradient) of the function at the current point.
                       If, instead, one takes steps proportional to the positive of the gradient, one approaches a local maximum of that function; the procedure is then known as gradient ascent.
    Parameters
    ----------
    configs (dict): configurations of the model

    Returns
    --------
    Train function (method/function): A train function of GA or GD that is defined in the ga/gd classes

    Example
    --------
    configs = {"type" : "genetic" }
    algorithm = get_algorithm(configs)

    """
    if configs["type"] == "gradient":
        return train_gd
    elif configs["type"] == "genetic":
        return train_ga
    else:
        assert (
            False
        ), "Unrecognised algorithm field in configs. It must have the value gradient or the value genetic."


def get_driver(
    configs: dict,
    info: dict = None,
    model_state_dict: collections.OrderedDict = None,
):
    """
    To get a driver object based on the configurations dictionary.
    The driver here are defined under the processor type tag in the configs dictionary and can be a
        SurrogateModel (Software processor) -  It is a deep neural network with information about the  control voltage ranges,
                        the amplification of the device and relevant noise simulations that it may have
        Hardware Processor - It establishes a connection (for a single, or multiple hardware DNPUs) with one of the following National Instruments measurement devices:
                * CDAQ-to-NiDAQ
                * CDAQ-to-CDAQ
                        * With a regular rack
                        * With a real time rack

    Parameters
    -----------
    configs (dict): configurations of the model

    Raises
    -------
    NotImplementedError: if configurations is not recognised

    Returns
    --------
    class object: Returns and driver object which can be a  CDAQtoCDAQ,CDAQtoNiDAQ or a SurrogateModel object

    Example
    --------
    configs = {"processsor_type" : "simulation_debug" }
    driver = get_driver(configs)

    """
    if configs["instrument_type"] == "cdaq_to_cdaq":
        return CDAQtoCDAQ(configs)
    elif configs["instrument_type"] == "cdaq_to_nidaq":
        return CDAQtoNiDAQ(configs)
    else:
        raise NotImplementedError(
            f"{configs['instrument_type']} 'instrument_type' configuration is not recognised. The simulation type has to be defined as 'cdaq_to_cdaq' or 'cdaq_to_nidaq'. "
        )
