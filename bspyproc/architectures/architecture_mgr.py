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
    if configs['processor_type'] == 'nn':
        raise NotImplementedError(f"{configs['processor_type']} 'processor_type' nn configuration is not implemented yet. ")
    if configs['processor_type'] == 'surrogate':
        return get_processor_architecture(configs)
    elif configs['processor_type'] == 'dnpu':
        return get_dnpu_architecture(configs).to(device=TorchUtils.get_accelerator_type())
    else:
        raise NotImplementedError(f"{configs['processor_type']} 'processor_type' configuration is not recognised. The simulation type has to be defined as 'nn', 'surrogate' or 'dpnu'. ")


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
