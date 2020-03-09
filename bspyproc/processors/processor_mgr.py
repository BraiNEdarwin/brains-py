from bspyproc.utils.pytorch import TorchUtils
from bspyproc.processors.simulation.dopanet import DNPU
from bspyproc.processors.simulation.surrogate import SurrogateModel
from bspyproc.processors.simulation.network import NeuralNetworkModel
from bspyproc.processors.hardware.setup_mgr import CDAQtoCDAQ, CDAQtoNiDAQ


def get_processor(configs):
    if configs['platform'] == 'hardware':
        return get_hardware_processor(configs)
    elif configs['platform'] == 'simulation':
        return get_simulation_processor(configs)
    else:
        raise NotImplementedError(f"Platform {configs['platform']} is not recognised. The platform has to be either 'hardware' or 'simulation'")


def get_hardware_processor(configs):
    if configs['processor_type'] == 'cdaq_to_cdaq':
        configs['input_instrument'] = 'cDAQ1Mod2'
        configs['output_instrument'] = 'cDAQ1Mod1'
        configs['trigger_source'] = 'cDAQ1'
        return CDAQtoCDAQ(configs)
    elif configs['processor_type'] == 'cdaq_to_nidaq':
        configs['input_instrument'] = 'dev1'
        configs['output_instrument'] = 'cDAQ1Mod1'
        return CDAQtoNiDAQ(configs)
    else:
        raise NotImplementedError(f"{configs['processor_type']} 'processor_type' configuration is not recognised. The simulation type has to be defined as 'cdaq_to_cdaq' or 'cdaq_to_nidaq'. ")


def get_simulation_processor(configs):
    if configs['processor_type'] == 'nn':
        return NeuralNetworkModel(configs)
    elif configs['processor_type'] == 'surrogate':
        return SurrogateModel(configs)
    elif configs['processor_type'] == 'dnpu':
        return DNPU(configs)
    else:
        raise NotImplementedError(f"{configs['processor_type']} 'processor_type' configuration is not recognised. The simulation type has to be defined as 'nn', 'surrogate' or 'dpnu'. ")