from bspyproc.utils.pytorch import TorchUtils
from bspyproc.architectures.multiplexing.simulation import TwoToOneDNPU, TwoToTwoToOneDNPU
from bspyproc.architectures.multiplexing.hardware import TwoToOneProcessor, TwoToTwoToOneProcessor


def get_architecture(configs):
    if configs['platform'] == 'hardware':
        return get_processor_architecture(configs)
    elif configs['platform'] == 'simulation':
        return get_simulation_architecture(configs)
    else:
        raise NotImplementedError(f"Platform {configs['platform']} is not recognised. The platform has to be either 'hardware' or 'simulation'")


def get_simulation_architecture(configs):
    if configs['simulation_type'] == 'neural_network':
        return get_neural_network_simulation_architecture(configs)
    elif configs['simulation_type'] == 'kinetic_monte_carlo':
        raise NotImplementedError(f"{configs['simulation_type']} 'simulation_type' configuration is not yet implemented.")
    else:
        raise NotImplementedError(f"{configs['simulation_type']} 'simulation_type' configuration is not recognised. The simulation type has to be defined as 'neural_network' or 'kinetic_monte_carlo'. ")


def get_neural_network_simulation_architecture(configs):
    if configs['network_type'] == 'device_model' or configs['network_type'] == 'nn_model':
        # raise NotImplementedError(f"{configs['network_type']} 'network_type' configuration is not implemented. ")
        return get_processor_architecture(configs)
    elif configs['network_type'] == 'dnpu':
        return get_dnpu_architecture(configs).to(device=TorchUtils.get_accelerator_type())
    else:
        raise NotImplementedError(f"{configs['network_type']} 'network_type' configuration is not recognised. The simulation type has to be defined as 'device_model', 'nn_model' or 'dpnu'. ")


def get_processor_architecture(configs):
    if configs['architecture_type'] == '21':
        return TwoToOneProcessor(configs)
    elif configs['architecture_type'] == '221':
        return TwoToTwoToOneProcessor(configs)
    else:
        raise NotImplementedError(f"Architecture type {configs['architecture_type']} is not recognised. The architecture_type has to be either '21' or '221' (as a String)")


def get_dnpu_architecture(configs):
    if configs['architecture_type'] == '21':
        return TwoToOneDNPU(configs)
    elif configs['architecture_type'] == '221':
        return TwoToTwoToOneDNPU(configs)
    else:
        raise NotImplementedError(f"Architecture type {configs['architecture_type']} is not recognised. The architecture_type has to be either '21' or '221' (as a String)")
