#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 17:18:34 2019
Trains a neural network given data. This trainer is intended for exploration of new designs of DNPU architectures,
so it is a convenience function. If you want to train standard tasks or the model, use the GD class in brainspy-algorithm package.
---------------
Arguments
data : List containing 2 tuples; the first with a training set (inputs,targets),
        the second with validation data. Both the inputs and targets must be 
        torch.Tensors (shape: nr_samplesXinput_dim, nr_samplesXoutput_dim).
network : The network to be trained
conf_dict : Configuration dictionary with hyper parameters for training
---------------
Returns:
???? network (torch.nn.Module) : trained network
costs (np.array)    : array with the costs (training,validation) per epoch

Notes:
    1) The dopantNet is composed by a surrogate model of a dopant network device
    and bias learnable parameters that serve as control inputs to tune the 
    device for desired functionality. If you have this use case, you can get the
    control voltage parameters via network.parameters():
        params = [p.clone().detach() for p in network.parameters()]
        control_voltages = params[0]
    2) For training the surrogate model, the outputs must be scaled by the 
    amplification. Hence, the output of the model and the errors are  NOT in nA.
    To get the errors in nA, scale by the amplification**2.
    The dopant network already outputs the prediction in nA. To get the output 
    of the surrogate model in nA, use the method .outputs(inputs).

@author: hruiz
"""

import numpy as np
# from bspyproc.utils.pytorch import TorchUtils
import torch


def trainer(network, training_data, validation_data=(None, None),
            loss_fn=torch.nn.MSELoss(), learning_rate=1e-2,
            nr_epochs=3000, batch_size=128, cv_penalty=0.5,
            save_dir='../../test/NN_test/',
            save_interval=10, **kwargs):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    network.to(device)
    # set configurations
    if "seed" in kwargs.keys():
        seed = kwargs["seed"]
        torch.manual_seed(seed)
        print(f'The torch RNG is seeded with {seed}')
    if "betas" in kwargs.keys():
        optimizer = torch.optim.Adam(network.parameters(),
                                     lr=learning_rate,
                                     betas=kwargs["betas"])
        print("Set betas to values: ", {kwargs["betas"]})
    else:
        optimizer = torch.optim.Adam(network.parameters(),
                                     lr=learning_rate)
    print('Prediction using ADAM optimizer')

    # Define variables
    x_train, y_train = training_data
    x_val, y_val = validation_data
    costs = np.zeros((nr_epochs, 2))  # training and validation costs per epoch
    if x_val is not None:
        samples = len(x_val)
    else:
        samples = len(x_train)

    for epoch in range(nr_epochs):

        network.train()
        permutation = torch.randperm(x_train.size()[0])  # Permute indices
        nr_minibatches = 0

        for i in range(0, len(permutation), batch_size):

            # Get prediction
            indices = permutation[i:i + batch_size]
            x = x_train[indices]
            y_pred = network(x)
            # GD step
            if 'regularizer' in dir(network):
                loss = loss_fn(
                    y_pred, y_train[indices]) + cv_penalty * network.regularizer()
            else:
                loss = loss_fn(y_pred, y_train[indices])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            nr_minibatches += 1

        network.eval()
        # Evaluate training error
        get_indices = torch.randperm(len(x_train))[:samples]
        x = x_train[get_indices]
        prediction = network(x)
        target = y_train[get_indices]
        costs[epoch, 0] = loss_fn(prediction, target).item()
        # assert not np.isnan(costs[epoch, 0]), "Loss is NaN!"
        # Evaluate Validation error
        if x_val is not None and y_val is not None:
            prediction = network(x_val)
            costs[epoch, 1] = loss_fn(prediction, y_val).item()
        else:
            costs[epoch, 1] = np.nan

        # if save_dir and epoch % save_interval == 0:
        #     save_model(network, save_dir+f'checkpoint_epoch{epoch}.pt')
        if np.isnan(costs[epoch, 0]):
            costs[-1, 0] = np.nan
            print('--------- Training interrupted value was Nan!! ---------')
            break
        if epoch % 300 == 0:
            print('Epoch:', epoch,
                  'Val. Error:', costs[epoch, 1],
                  'Training Error:', costs[epoch, 0])

    return costs


def save_model(model, path):
    """
    Saves the model in given path, all other attributes are saved under
    the 'info' key as a new dictionary.
    """
    model.eval()
    state_dic = model.state_dict()
    if 'info' in dir(model):
        state_dic['info'] = model.info
    torch.save(state_dic, path)


if __name__ == '__main__':

    import torch
    import torch.nn as nn
    import numpy as np
    import matplotlib.pyplot as plt
    from bspyalgo.utils.io import load_configs
    from bspyproc.architectures.dnpu.modules import DNPU_Layer
    from bspyproc.utils.pytorch import TorchUtils

    # Generate model
    NODE_CONFIGS = load_configs('configs/configs_nn_model.json')
    nr_nodes = 5
    input_list = [[0, 3, 4]] * nr_nodes
    data_dim = 20
    linear_layer = nn.Linear(data_dim, len(input_list[0]) * nr_nodes).to(device=TorchUtils.get_accelerator_type())
    dnpu_layer = DNPU_Layer(input_list, NODE_CONFIGS)
    model = nn.Sequential(linear_layer, dnpu_layer)

    # Generate data
    nr_train_samples = 50
    nr_val_samples = 10
    x = TorchUtils.format_tensor(torch.rand(nr_train_samples + nr_val_samples, data_dim))
    y = TorchUtils.format_tensor(5. * torch.ones(nr_train_samples + nr_val_samples, nr_nodes))

    inp_train = x[:nr_train_samples]
    t_train = y[:nr_train_samples]
    inp_val = x[:nr_val_samples]
    t_val = y[:nr_val_samples]

    node_params_start = [p.clone().cpu().detach() for p in model.parameters() if not p.requires_grad]
    learnable_params_start = [p.clone().cpu().detach() for p in model.parameters() if p.requires_grad]

    costs = trainer(model, (inp_train, t_train), validation_data=(inp_val, t_val),
                    nr_epochs=5000,
                    batch_size=len(t_train),
                    learning_rate=3e-5,
                    save_dir='test/dnpu_arch_test/',
                    save_interval=np.inf)

    out = model(inp_val)
    print(f'OUTPUT: {out.data.cpu().numpy()}')

    node_params_end = [p.clone().cpu().detach() for p in model.parameters() if not p.requires_grad]
    learnable_params_end = [p.clone().cpu().detach() for p in model.parameters() if p.requires_grad]

    print("CV params at the beginning: \n ", learnable_params_start[2:])
    print("CV params at the end: \n", learnable_params_end[2:])
    abs_diff_cv_params = [np.sum(np.abs(b.numpy() - a.numpy())) for b, a in zip(learnable_params_start, learnable_params_end)]
    print(f'Abs. difference between CV parameters before-after: {sum(abs_diff_cv_params)}')

    print("Example node params at the beginning: \n", node_params_start[1])
    print("Example node params at the end: \n", node_params_end[1])
    abs_diff_node_params = [np.sum(np.abs(b.numpy() - a.numpy())) for b, a in zip(node_params_start, node_params_end)]
    print(f'Abs. difference between node parameters before-after: {sum(abs_diff_node_params)}')

    plt.figure()
    plt.plot(costs)
    plt.title("Loss per epoch")
    plt.legend(["Training", "Validation"])
    plt.show()
