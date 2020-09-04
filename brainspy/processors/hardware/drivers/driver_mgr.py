from brainspy.utils.pytorch import TorchUtils

# from brainspy.processors.dnpu import DNPU
from brainspy.processors.simulation.surrogate import SurrogateModel

# from brainspy.processors.simulation.network import NeuralNetworkModel
from brainspy.processors.hardware.drivers.setups import CDAQtoCDAQ, CDAQtoNiDAQ


def get_driver(configs):
    if configs["processor_type"] == "cdaq_to_cdaq":
        # configs['input_instrument'] = 'cDAQ1Mod2'
        # configs['output_instrument'] = 'cDAQ1Mod1'
        # configs['trigger_source'] = 'cDAQ1'
        return CDAQtoCDAQ(configs)
    elif configs["processor_type"] == "cdaq_to_nidaq":
        # configs['input_instrument'] = 'dev1'
        # configs['output_instrument'] = 'cDAQ1Mod1'
        return CDAQtoNiDAQ(configs)
    elif configs["processor_type"] == "simulation_debug":
        return SurrogateModel(configs)
    else:
        raise NotImplementedError(
            f"{configs['processor_type']} 'processor_type' configuration is not recognised. The simulation type has to be defined as 'cdaq_to_cdaq' or 'cdaq_to_nidaq'. "
        )
