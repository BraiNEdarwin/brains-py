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
import os
from tqdm import trange
from bspyalgo.utils.io import create_directory_timestamp, save_pickle, save_configs


def save_model(model, path, name):
    """
    Saves the model in given path.
    """
    state_dic = model.state_dict()
    file_path = os.path.join(path, name + '.pt')
    torch.save(state_dic, file_path)


def save_parameters_as_numpy(model, path):
    parameters = {k: v.cpu().detach().numpy()
                  for k, v in model.named_parameters() if v.requires_grad}
    save_pickle(parameters, os.path.join(path, 'learned_parameters.dat'))


def batch_generator(data, batch_size):
    nr_samples = len(data[0])
    permutation = torch.randperm(nr_samples)  # Permute indices
    i = 0
    while i < nr_samples:
        indices = permutation[i:i + batch_size]
        yield data[0][indices], data[1][indices]
        i += batch_size


def batch_training(batch_iter, network, optimizer, loss_fn, config_dict):
    for _, batch in enumerate(batch_iter):
        # Get prediction
        y_pred = network(batch[0].to('cuda', non_blocking=True))
        y_targets = batch[1].to('cuda', non_blocking=True)
        # GD step
        if 'regularizer' in dir(network):
            loss = loss_fn(y_pred, y_targets) + config_dict["cv_penalty"] * network.regularizer()  # usually set to cv_penalty=0.5
        else:
            loss = loss_fn(y_pred, y_targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def set_optimizer(network, config_dict):
    print('Prediction using ADAM optimizer')
    if "seed" in config_dict.keys():
        seed = config_dict["seed"]
        torch.manual_seed(seed)
        print(f'The torch RNG is seeded with {seed}')
    if "betas" in config_dict.keys():
        print("Set betas to values: ", {config_dict["betas"]})
        return torch.optim.Adam(network.parameters(),
                                lr=config_dict["learning_rate"],
                                betas=config_dict["betas"])
    else:
        return torch.optim.Adam(network.parameters(),
                                lr=config_dict["learning_rate"])


def accuracy(output, targets):
    predicted_labels = torch.argmax(output, 1).data.detach().numpy()
    return sum(targets.data.detach().numpy() == predicted_labels) / len(targets)


def train_with_generator(network, training_data, config_dict, validation_data=None,
                         loss_fn=torch.nn.MSELoss()):
    print('------- TRAINING WITH GENERATOR ---------')
    # set configurations
    optimizer = set_optimizer(network, config_dict)
    config_dict["save_dir"] = create_directory_timestamp(config_dict["save_dir"], config_dict["name"])
    # Define variables
    costs = np.zeros((config_dict["nr_epochs"], 2))  # training and validation costs per epoch

    for epoch in range(config_dict["nr_epochs"]):
        network.train()
        batch_training(batch_generator(training_data, config_dict["batch_size"]),
                       network, optimizer, loss_fn, config_dict)
        network.eval()

        # Evaluate training error
        train_batch = next(batch_generator(training_data, len(training_data[0])))
        prediction = network(train_batch[0])
        costs[epoch, 0] = loss_fn(prediction, train_batch[1]).item()
        # Evaluate Validation error
        if validation_data is not None:
            prediction = network(validation_data[0])
            costs[epoch, 1] = loss_fn(prediction, validation_data[1]).item()
        else:
            costs[epoch, 1] = np.nan
        if 'save_interval' in config_dict.keys() and epoch % config_dict["save_interval"] == 0:
            save_model(network, config_dict["save_dir"], f'checkpoint_epoch{epoch}')
        if np.isnan(costs[epoch, 0]):
            costs[-1, 0] = np.nan
            print('--------- Training interrupted value was Nan!! ---------')
            break
        if epoch % 100 == 0:
            print('Epoch:', epoch,
                  'Val. Error:', costs[epoch, 1],
                  'Training Error:', costs[epoch, 0])

    return costs


def train_with_loader(network, training_data, config_dict, validation_data=None,
                      loss_fn=torch.nn.MSELoss()):
    print('------- TRAINING WITH LOADER ---------')
    # set configurations
    optimizer = set_optimizer(network, config_dict)
    config_dict["save_dir"] = create_directory_timestamp(config_dict["save_dir"], config_dict["name"])
    save_configs(config_dict, os.path.join(config_dict["save_dir"], "training_configs.json"))
    # Define variables
    costs = np.zeros((config_dict["nr_epochs"], 2))  # training and validation costs per epoch
    accuracy_over_epochs = np.zeros((config_dict["nr_epochs"], 2))
    looper = trange(config_dict["nr_epochs"], desc='Initialising')
    for epoch in looper:
        network.train()
        batch_training(training_data, network, optimizer, loss_fn, config_dict)
        network.eval()

        # Evaluate training error
        training_losses = [loss_fn(network(x.to('cuda', non_blocking=True)).cpu(), y).item() for x, y in training_data]
        costs[epoch, 0] = np.mean(training_losses)
        accuracy_over_epochs[epoch, 0] = np.mean([accuracy(network(x.to('cuda', non_blocking=True)).cpu(), y).item() for x, y in training_data])
        # Evaluate Validation error
        if validation_data is not None:
            data, targets = iter(validation_data).next()
            prediction = network(data.squeeze().to('cuda', non_blocking=True)).cpu()
            costs[epoch, 1] = loss_fn(prediction, targets).item()
            accuracy_over_epochs[epoch, 1] = accuracy(prediction, targets).item()
        else:
            costs[epoch, 1] = np.nan

        if 'save_interval' in config_dict.keys() and epoch % config_dict["save_interval"] == 10:
            save_model(network, config_dict["save_dir"], f'checkpoint_{epoch}')
        if np.isnan(costs[epoch, 0]):
            costs[-1, 0] = np.nan
            print('--------- Training interrupted value was Nan!! ---------')
            break
        looper.set_description(
            f' Epoch: {epoch} | Training Error:{costs[epoch, 0]:7.3f} | Val. Error:{costs[epoch, 1]:7.3f}')

    save_model(network, config_dict["save_dir"], 'final_model')
    save_parameters_as_numpy(network, config_dict["save_dir"])
    np.savez(os.path.join(config_dict["save_dir"], 'training_history'), costs=costs, accuracy=accuracy)
    return costs, accuracy_over_epochs


if __name__ == '__main__':

    import torch
    import torch.nn as nn
    import matplotlib.pyplot as plt
    from bspyalgo.utils.io import load_configs
    from bspyproc.architectures.dnpu.modules import DNPU_Layer
    from bspyproc.utils.pytorch import TorchUtils

    # Generate model
    NODE_CONFIGS = load_configs('/home/hruiz/Documents/PROJECTS/DARWIN/Code/packages/brainspy/brainspy-processors/configs/configs_nn_model.json')
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
    inp_val = x[nr_train_samples:]
    t_val = y[nr_train_samples:]

    node_params_start = [p.clone().cpu().detach() for p in model.parameters() if not p.requires_grad]
    learnable_params_start = [p.clone().cpu().detach() for p in model.parameters() if p.requires_grad]
    costs = train_with_generator(model, (inp_train, t_train), validation_data=(inp_val, t_val),
                                 nr_epochs=3000,
                                 batch_size=int(len(t_train) / 10),
                                 learning_rate=3e-3,
                                 save_dir='test/dnpu_arch_test/',
                                 save_interval=np.inf)

    model.eval()
    out_val = model(inp_val).cpu().detach().numpy()
    out_train = model(inp_train).cpu().detach().numpy()

    plt.figure()
    plt.hist(out_train.flatten())
    plt.hist(out_val.flatten())
    plt.show()

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
