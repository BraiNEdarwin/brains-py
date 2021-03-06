from brainspy.processors.hardware.drivers.nidaq import CDAQtoNiDAQ
from brainspy.processors.hardware.drivers.cdaq import CDAQtoCDAQ
from brainspy.processors.simulation.processor import SurrogateModel
import torch

import brainspy.algorithms.modules.signal as criterion
import brainspy.algorithms.modules.optim as bspyoptim
from brainspy.algorithms.ga import train as train_ga
from brainspy.algorithms.gd import train as train_gd

from brainspy.utils.pytorch import TorchUtils


def get_criterion(configs: dict):  # can also return a tensor
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
    The function returns an optimizer object which include added information to train a specific model.

    Parameters
    ----------
        model (nn.Module object): An nn.Module object which can be a DNPU,Processor or a Surrogate Model object
        configs (dict): configurations for the model

    Returns
    -------
        class object: Returns and optimizer object which can be a GeneticOptimizer or an Adam optimizer
    """
    if configs["optimizer"] == "genetic":
        # TODO: get gene ranges from model
        if "gene_range" in configs:
            return bspyoptim.GeneticOptimizer(
                configs["gene_range"], configs["partition"], configs["epochs"]
            )
        else:
            return bspyoptim.GeneticOptimizer(
                model.get_control_ranges(), configs["partition"], configs["epochs"]
            )
    elif configs["optimizer"] == "elm":
        print("ELM optimizer not implemented yet")
        # return get_adam(parameters, configs)
    elif configs["optimizer"] == "adam":
        return get_adam(model, configs)
    else:
        assert False, "Optimiser name {configs['optimizer']} not recognised. Please try"


def get_adam(model: object, configs: dict):
    """The function returns an Adam optimizer object which include added information to train a specific model.

    Parameters
    ----------
        model (nn.Module object): An nn.Module object which can be a DNPU,Processor or a SurrogateModel object
        configs (dict): configurations of the model

    Returns
    -------
        class object: Returns and optimizer Adam optimizer object
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
    """The function returns a train function, either GA or GD based on the configurations dictionary

    Parameters
    ----------
        configs (dict): configurations of the model

    Returns
    --------
        Train function (method/function): A train function of GA or GD that is defined in the ga/gd classes
    """
    if configs["type"] == "gradient":
        return train_gd
    elif configs["type"] == "genetic":
        return train_ga
    else:
        assert (
            False
        ), "Unrecognised algorithm field in configs. It must have the value gradient or the value genetic."


def get_driver(configs: dict):
    """The function returns a driver object based on the configurations dictionary

    Parameters
    -----------
        configs (dict): configurations of the model

    Raises:
        NotImplementedError: if configurations is not recognised

    Returns:
        class object: Returns and driver object which can be a  CDAQtoCDAQ,CDAQtoNiDAQ or a SurrogateModel object
    """
    if configs["processor_type"] == "cdaq_to_cdaq":
        return CDAQtoCDAQ(configs)
    elif configs["processor_type"] == "cdaq_to_nidaq":
        return CDAQtoNiDAQ(configs)
    elif configs["processor_type"] == "simulation_debug":
        return SurrogateModel(configs)
    else:
        raise NotImplementedError(
            f"{configs['processor_type']} 'processor_type' configuration is not recognised. The simulation type has to be defined as 'cdaq_to_cdaq' or 'cdaq_to_nidaq'. "
        )
