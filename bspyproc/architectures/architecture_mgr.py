from bspyproc.architectures.simulation.dnpu_21 import TwoToOneDNPU
from bspyproc.architectures.simulation.dnpu_221 import TwoToTwoToOneDNPU


def get_architecture(configs):
    if configs['platform'] == 'hardware':
        return get_hardware_architecture(configs)
    elif configs['platform'] == 'simulation':
        return get_simulation_architecture(configs)
    else:
        raise NotImplementedError(f"Platform {configs['platform']} is not recognised. The platform has to be either 'hardware' or 'simulation'")


def get_hardware_architecture(configs):
    if configs['setup_type'] == 'cdaq_to_cdaq':
        configs['input_instrument'] = 'cDAQ1Mod2'
        configs['output_instrument'] = 'cDAQ1Mod1'
        configs['trigger_source'] = 'cDAQ1'
        # return CDAQtoCDAQ(configs)
    elif configs['setup_type'] == 'cdaq_to_nidaq':
        configs['input_instrument'] = 'dev1'
        configs['output_instrument'] = 'cDAQ1Mod1'
        # return CDAQtoNiDAQ(configs)
    else:
        raise NotImplementedError(f"{configs['setup_type']} 'setup_type' configuration is not recognised. The simulation type has to be defined as 'cdaq_to_cdaq' or 'cdaq_to_nidaq'. ")


def get_simulation_architecture(configs):
    if configs['simulation_type'] == 'neural_network':
        return get_neural_network_simulation_architecture(configs)
    elif configs['simulation_type'] == 'kinetic_monte_carlo':
        raise NotImplementedError(f"{configs['simulation_type']} 'simulation_type' configuration is not yet implemented.")
    else:
        raise NotImplementedError(f"{configs['simulation_type']} 'simulation_type' configuration is not recognised. The simulation type has to be defined as 'neural_network' or 'kinetic_monte_carlo'. ")


def get_neural_network_simulation_architecture(configs):
    if configs['network_type'] == 'device_model':
        # return TorchModel(configs['torch_model_dict'])
        raise NotImplementedError(f"{configs['network_type']} 'network_type' configuration is not yet implemented. ")
    elif configs['network_type'] == 'nn_model':
        # return TorchModel(configs['torch_model_dict'])
        raise NotImplementedError(f"{configs['network_type']} 'network_type' configuration is not yet implemented. ")
    elif configs['network_type'] == 'dnpu':
        return get_dnpu_architecture(configs)
    else:
        raise NotImplementedError(f"{configs['network_type']} 'network_type' configuration is not recognised. The simulation type has to be defined as 'device_model', 'nn_model' or 'dpnu'. ")


def get_dnpu_architecture(configs):
    if configs['architecture_type'] == '21':
        return TwoToOneDNPU(configs)
    elif configs['architecture_type'] == '221':
        return TwoToTwoToOneDNPU(configs)
    else:
        raise NotImplementedError(f"Architecture type {configs['architecture_type']} is not recognised. The architecture_type has to be either '21' or '221' (as a String)")
