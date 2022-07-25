"""
Class to retrieve data specified in a dictionary used to train a  model.
The class methods require a configuration dictionary with data keys specific to do a task.
It can be used to get an optimizer,fitness function,driver or an algorithm for training a model.
"""

import torch
from brainspy.processors.hardware.drivers.nidaq import CDAQtoNiDAQ
from brainspy.processors.hardware.drivers.cdaq import CDAQtoCDAQ
import brainspy.utils.signal as criterion
import brainspy.algorithms.ga as bspyoptim
from brainspy.algorithms.ga import train as train_ga
from brainspy.algorithms.gd import train as train_gd


def get_criterion(name: str):
    """
    Returns a loss/fitness function from brainspy.algorithms.modules.signal module given a string
    name.

    Parameters
    ----------
    name : Name of the criterion that will be instantiated from
           brainspy.algorithms.modules.signal

    Returns
    -------
    A method from brainspy.algorithms.modules.signal containing either a loss or a fitness function.

    Example
    --------

    criterion = get_criterion("corr_fit")

    """
    if name == "accuracy_fit":
        return criterion.accuracy_fit
    elif name == "corrsig":
        return criterion.corrsig
    elif name == "corr_fit":
        return criterion.corr_fit
    elif name == "corrsig_fit":
        return criterion.corrsig_fit
    elif name == "fisher":
        return criterion.fisher
    elif name == "fisher_fit":
        return criterion.fisher_fit
    elif name == "bce":
        return torch.nn.BCEWithLogitsLoss()
    elif name == "sigmoid_nn_distance":
        return criterion.sigmoid_nn_distance
    else:
        raise NotImplementedError(f"Criterion {name} is not recognised.")


def get_optimizer(model: object, configs: dict):
    """
    Gets either a genetic algorithm or a gradient descent pytorch optimizer object from a
    dictionary.

    Parameters
    ----------
    Model (nn.Module object): An nn.Module object which can also be a DNPU, Processor or
    a SurrogateModel. On the gradient descent, it is required to gather the learnable
    parameters from the model. On the genetic algorithm, it is required to gather the
    control ranges.

    configs : dict
            This configuration is different depending on whether a genetic or
            a gradient descent optimiser is requested.
                Gradient descent keys: See the function get_adam
                Genetic algorithm keys:
                    * Gene range (Optional): Specifies what is the range of the control
                                                electrodes. If this key is not present, the
                                                gene range will be calculated automatically
                                                from the control electrode range function
                                                of the model.
                    * Partition: Tuple[int, int] Defines the partition of genomes when
                        generating offspring.
                    * Epochs: Number of loops that the algorithm is going to take.

    Returns
    -------
    Returns an object which can be a brainspy.algorithms.optim.GeneticOptimizer or
    an torch.optim.Adam optimizer

    Example
    --------
    configs = {"optimizer" : "genetic",
               "partition": [4,22],
               "epochs": 100}

    model = CustomDNPUModel()

    optimizer = get_optimizer(model,configs)

    -------
    configs = {"optimizer" : "adam",
            "learning_rate": 1e-3}

    model = torch.nn.Linear(1,1)

    optimizer = get_optimizer(model,configs)

    """
    if configs["optimizer"] == "genetic":
        # TODO: get gene ranges from model
        if "gene_range" in configs:
            return bspyoptim.GeneticOptimizer(configs["gene_range"],
                                              configs["partition"],
                                              configs["epochs"])
        else:
            # Only a single device is supported, therefore model.get_control_ranges()[0]
            return bspyoptim.GeneticOptimizer(
                model.get_control_ranges()[0],  # type: ignore[attr-defined]
                configs["partition"],
                configs["epochs"])
    elif configs["optimizer"] == "adam":
        return get_adam(model, configs)
    else:
        assert False, "Optimiser name {configs['optimizer']} not recognised. Please try"


def get_adam(model: object, configs: dict = {}):
    """
    To get an Adam optimizer object which include added information to train a specific model.
    It is for first-order gradient-based optimization of stochastic objective functions, based
    on adaptive estimates of lower-order moments.

    Parameters
    ----------
    model : torch.nn.Module
        A Module object which can be a DNPU,Processor or a SurrogateModel
        object.
    configs (dict): Configurations of the adam optimizer. The configurations do not require to have
                    all of these keys. The keys of the dictionary are as follows:
                        * learning_rate: float
                        * betas: Tuple[float, float]
                        * epsilon: float
                        * weight_decay: float
                        * amsgrad: boolean
            More information on what these keys do can be found at:
            https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam

    Returns
    -------
    class object: Returns and optimizer Adam optimizer object.

    Example
    --------
    configs = {"learning_rate": 0.0001}
    model = torch.nn.Linear(1,1)
    optimizer = get_adam(model,configs)

    """
    # Initialise parameters
    parameters = filter(lambda p: p.requires_grad,
                        model.parameters())  # type: ignore[attr-defined]

    # Initialise dictionary configs
    if 'learning_rate' in configs:
        lr = configs['learning_rate']
    else:
        lr = 1e-3
    if 'betas' in configs:
        betas = configs['betas']
    else:
        betas = (0.9, 0.999)
    if 'eps' in configs:
        eps = configs['eps']
    else:
        eps = 1e-8
    if 'weight_decay' in configs:
        weight_decay = configs['weight_decay']
    else:
        weight_decay = 0
    if 'amsgrad' in configs:
        amsgrad = configs['amsgrad']
    else:
        amsgrad = False

    return torch.optim.Adam(parameters,
                            lr=lr,
                            betas=betas,
                            eps=eps,
                            weight_decay=weight_decay,
                            amsgrad=amsgrad)


def get_algorithm(name: str):
    """
    To get a default train function for either GA - genetic algorithm or GD - Gradient Descent,
    based on its name.
    Genetic Algorithm : In computer science and operations research, a genetic algorithm (GA) is a
                        meta-heuristic inspired by the process of natural selection that belongs to
                        the larger class of evolutionary algorithms (EA). Genetic algorithms are
                        commonly used to generate high-quality solutions to optimization and search
                        problems by relying on bio-inspired operators such as mutation, crossover
                        and selection. This algorithm is suitable for experiments with reservoir
                        computing.

    Gradient Descent : Gradient descent is a first-order iterative optimization algorithm for
                       finding the minimum of a function. To find a local minimum of a function
                       using gradient descent, one takes steps proportional to the negative of
                       the gradient (or approximate gradient) of the function at the current point.

    Parameters
    ----------
    name : str
        Name of the algorithm. The string value can either be 'gradient' or 'genetic'.

    Returns
    --------
    A method containting a default training function for GA or GD as defined in from
    brainspy.algorithms.ga/gd.

    Example
    --------
    algorithm = get_algorithm('genetic')

    --------
    algorithm = get_algorithm('gradient')

    """
    if name == "gradient":
        return train_gd
    elif name == "genetic":
        return train_ga
    else:
        raise NotImplementedError(
            "Unrecognised algorithm field in configs." +
            " It must have the value gradient or the value genetic.")


def get_driver(configs: dict):
    """
    To get an instance of a driver object from brainspy.processors.hardware.drivers.nidaq/cdaq
    based on a configurations dictionary.
    The driver here are defined under the processor type tag in the configs dictionary and can be a
        SurrogateModel (Software processor) -  It is a deep neural network with information about
                        the control voltage ranges, the amplification of the device and relevant
                        noise simulations that it may have.
        Hardware Processor - It establishes a connection (for a single, or multiple hardware DNPUs)
                             with one of the following National Instruments measurement devices.
                * CDAQ-to-NiDAQ
                * CDAQ-to-CDAQ
                        * With a regular rack
                        * With a real time rack

    Parameters
    -----------
    configs : dict
        Configurations of the model.

    Raises
    -------
    NotImplementedError: If configurations is not recognised.

    Returns
    --------
    brainspy.processors.hardware.drivers.ni.NationalInstrumentsSetup: Returns and driver object
    which can be CDAQtoCDAQ or CDAQtoNiDAQ.

    Example to load a CDAQtoNiDAQ driver
    (differnt configurations can be provided for differt drivers)
    --------
        configs = {}
        configs["processor_type"] = "cdaq_to_nidaq"
        configs["input_indices"] = [2, 3]
        configs["electrode_effects"] = {}
        configs["electrode_effects"]["amplification"] = 3
        configs["electrode_effects"]["clipping_value"] = [-300, 300]
        configs["electrode_effects"]["noise"] = {}
        configs["electrode_effects"]["noise"]["noise_type"] = "gaussian"
        configs["electrode_effects"]["noise"]["variance"] = 0.6533523201942444
        configs["driver"] = {}

        configs["driver"]["instruments_setup"] = {}
        configs["driver"]["instruments_setup"]["multiple_devices"] = False
        configs["driver"]["instruments_setup"]["trigger_source"] = "cDAQ1/segment1"
        configs["driver"]["instruments_setup"]["activation_instrument"] = "cDAQ1Mod3"
        configs["driver"]["instruments_setup"]["activation_sampling_frequency"] = 1000
        configs["driver"]["instruments_setup"]["activation_channels"] = [
            0,
            2,
            5,
            3,
            4,
            6,
            1,
        ]
        configs["driver"]["instruments_setup"]["activation_voltages"] = [
            [-1.2, 0.6],
            [-1.2, 0.6],
            [-1.2, 0.6],
            [-1.2, 0.6],
            [-1.2, 0.6],
            [-0.7, 0.3],
            [-0.7, 0.3],
        ]
        configs["driver"]["instruments_setup"]["readout_instrument"] = "cDAQ1Mod4"
        configs["driver"]["instruments_setup"]["readout_sampling_frequency"] = 1000
        configs["driver"]["instruments_setup"]["readout_channels"] = [
            4
        ]
        configs["waveform"] = {}
        configs["waveform"]["plateau_length"] = 10
        configs["waveform"]["slope_length"] = 30
        driver = get_driver(configs)

    """
    if configs["instrument_type"] == "cdaq_to_cdaq":
        return CDAQtoCDAQ(configs)
    elif configs["instrument_type"] == "cdaq_to_nidaq":
        return CDAQtoNiDAQ(configs)
    else:
        raise NotImplementedError(
            f"{configs['instrument_type']} 'instrument_type' configuration is not recognised."
            +
            " The simulation type has to be defined as 'cdaq_to_cdaq' or 'cdaq_to_nidaq'."
        )
