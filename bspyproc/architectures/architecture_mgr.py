def get_architecture(configs):
    if configs['platform'] == 'hardware':
        return get_hardware_processor(configs)
    elif configs['platform'] == 'simulation':
        return get_simulation_processor(configs)
    else:
        raise NotImplementedError(f"Platform {configs['platform']} is not recognised. The platform has to be either 'hardware' or 'simulation'")


def get_hardware_processor(configs):
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


def get_simulation_processor(configs):
    if configs['simulation_type'] == 'neural_network':
        return get_neural_network_simulation_processor(configs)
    elif configs['simulation_type'] == 'kinetic_monte_carlo':
        # return SimulationKMC()
    else:
        raise NotImplementedError(f"{configs['simulation_type']} 'simulation_type' configuration is not recognised. The simulation type has to be defined as 'neural_network' or 'kinetic_monte_carlo'. ")
