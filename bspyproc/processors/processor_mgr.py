from bspyproc.utils.pytorch import TorchUtils
from bspyproc.processors.simulation.dopanet import DNPU
from bspyproc.processors.simulation.network import TorchModel
from bspyproc.processors.hardware.setup_mgr import CDAQtoCDAQ, CDAQtoNiDAQ


def get_processor(configs):
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
        return CDAQtoCDAQ(configs)
    elif configs['setup_type'] == 'cdaq_to_nidaq':
        configs['input_instrument'] = 'dev1'
        configs['output_instrument'] = 'cDAQ1Mod1'
        return CDAQtoNiDAQ(configs)
    else:
        raise NotImplementedError(f"{configs['setup_type']} 'setup_type' configuration is not recognised. The simulation type has to be defined as 'cdaq_to_cdaq' or 'cdaq_to_nidaq'. ")


def get_simulation_processor(configs):
    if configs['simulation_type'] == 'neural_network':
        return get_neural_network_simulation_processor(configs).to(device=TorchUtils.get_accelerator_type())
    else:
        raise NotImplementedError(f"{configs['simulation_type']} 'simulation_type' configuration is not recognised. The simulation type has to be defined as 'neural_network' or 'kinetic_monte_carlo'. ")


def get_neural_network_simulation_processor(configs):
    if configs['network_type'] == 'device_model' or configs['network_type'] == 'nn_model':
        return TorchModel(configs)
    elif configs['network_type'] == 'dnpu':
        return DNPU(configs)
    else:
        raise NotImplementedError(f"{configs['network_type']} 'network_type' configuration is not recognised. The simulation type has to be defined as 'device_model', 'nn_model' or 'dpnu'. ")


if __name__ == '__main__':
    PROCESSOR_CONFIGS = {}
    PROCESSOR_CONFIGS['platform'] = 'simulation'
    PROCESSOR_CONFIGS['simulation_type'] = 'neural_network'
    PROCESSOR_CONFIGS['network_type'] = 'raw_model'
    PROCESSOR_CONFIGS['input_indices'] = [0, 4]
    PROCESSOR_CONFIGS['torch_model_path'] = 'tmp/inputs/test_model/checkpoint3000_02-07-23h47m.pt'

    import matplotlib.pyplot as plt
    import torch
    from torch import nn
    import numpy as np
    x = 0.5 * np.random.randn(10, 7)
    # x = torch.tensor(x).to(DEVICE)
    x = TorchUtils.get_tensor_from_numpy(x)
    # target = torch.tensor([[5]] * 10).to(DEVICE)
    target = TorchUtils.get_tensor_from_list([[5]] * 10)
    node = get_processor(PROCESSOR_CONFIGS)  # ([0, 4])
    loss = nn.MSELoss()
    optimizer = torch.optim.SGD([{'params': node.parameters()}], lr=0.00005)

    LOSS_LIST = []
    CHANGE_PARAMS_NET = []
    CHANGE_PARAMS0 = []

    START_PARAMS = [p.clone().detach() for p in node.parameters()]

    for eps in range(2000):

        optimizer.zero_grad()
        out = node(x)
        if np.isnan(out.data.cpu().numpy()[0]):
            break
        LOSS = loss(out, target)  # + node.regularizer()
        LOSS.backward()
        optimizer.step()
        LOSS_LIST.append(LOSS.data.cpu().numpy())
        CURRENT_PARAMS = [p.clone().detach() for p in node.parameters()]
        DELTA_PARAMS = [(current - start).sum() for current, start in zip(CURRENT_PARAMS, START_PARAMS)]
        CHANGE_PARAMS0.append(DELTA_PARAMS[0])
        CHANGE_PARAMS_NET.append(sum(DELTA_PARAMS[1:]))

    END_PARAMS = [p.clone().detach() for p in node.parameters()]
    print("CV params at the beginning: \n ", START_PARAMS[0])
    print("CV params at the end: \n", END_PARAMS[0])
    print("Example params at the beginning: \n", START_PARAMS[-1][:8])
    print("Example params at the end: \n", END_PARAMS[-1][:8])
    print("Length of elements in node.parameters(): \n", [len(p) for p in END_PARAMS])
    print("and their shape: \n", [p.shape for p in END_PARAMS])
    print(f'OUTPUT: \n {out.data.cpu()}')

    plt.figure()
    plt.plot(LOSS_LIST)
    plt.title("Loss per epoch")
    plt.show()
    plt.figure()
    plt.plot(CHANGE_PARAMS0)
    plt.plot(CHANGE_PARAMS_NET)
    plt.title("Difference of parameter with initial params")
    plt.legend(["CV params", "Net params"])
    plt.show()
