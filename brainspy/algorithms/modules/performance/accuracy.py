#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 11:42:27 2018

@author: hruiz and ualegre
"""


import os
import torch
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt
from torch.utils.data import random_split, SubsetRandomSampler

from brainspy.algorithms.modules.performance.perceptron import Perceptron, PerceptronDataset
from brainspy.utils.pytorch import TorchUtils


def get_accuracy(inputs, targets, configs=None, node=None):
    # Assumes that the input_waveform and the target_waveform have the shape (n_total,1)
    # Normalizes the data; it is assumed that the target_waveform has binary values

    assert len(inputs.shape) != 1 and len(targets.shape) != 1, "Please unsqueeze inputs and targets"

    if configs is None:
        configs = get_default_node_configs()
    # Initialise perceptron
    if node is None:
        train = True
        node = Perceptron()
    else:
        train = False

    # Initialise results dictionary
    results = {}
    results['inputs'] = inputs.clone()
    std = inputs.std(axis=0)
    if std == 0:  # This is to avoid nan values when normalising the input.
        std = 1
    results['norm_inputs'] = (inputs - inputs.mean(axis=0)) / std
    results['targets'] = targets

    if train:
        # Prepare perceptron data
        dataset = PerceptronDataset(results['norm_inputs'], results['targets'])
        lengths = [len(dataset) * configs['split'][0], len(dataset) * configs['split'][1]]
        dataloaders = random_split(dataset, lengths)
        # If there is no validation dataloader, remove it
        if len(dataloaders[1]) == 0:
            del dataloaders[1]
        # Train the perceptron
        accuracy, predicted_labels, threshold, node = train_perceptron(dataloaders, configs, node)
    else:
        accuracy, predicted_labels = evaluate_accuracy(results['norm_inputs'], results['targets'], node)
        threshold = get_decision_boundary(node)
        print('Best accuracy: ' + str(accuracy.item()))

    # Save remaining results dictionary
    # results['predictions'] = predictions
    results['threshold'] = threshold
    results['predicted_labels'] = predicted_labels
    results['node'] = node
    results['accuracy_value'] = accuracy

    return results


def get_default_node_configs():
    configs = {}
    configs['epochs'] = 100
    configs['learning_rate'] = 0.0007
    configs['betas'] = [0.999, 0.999]
    configs['split'] = [1, 0]
    configs['mini_batch'] = 256
    return configs


def train_perceptron(dataloaders, configs, node=None):
    # Initialise key elements of the trainer
    node = TorchUtils.format_tensor(node)
    loss = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(node.parameters(), lr=configs['learning_rate'], betas=configs['betas'])
    best_accuracy = -1
    looper = trange(configs['epochs'], desc='Calculating accuracy')

    for epoch in looper:
        for inputs, targets in dataloaders[0]:
            optimizer.zero_grad()
            predictions = node(inputs)
            cost = loss(predictions, targets)
            cost.backward()
            optimizer.step()
        with torch.no_grad():
            inputs, targets = dataloaders[get_index(dataloaders)].dataset[:]
            accuracy, predicted_labels = evaluate_accuracy(inputs, targets, node)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                decision_boundary = get_decision_boundary(node)
                # TODO: Add a more efficient stopping mechanism ?
                if best_accuracy >= 100.:
                    looper.set_description(f'Reached 100/% accuracy. Stopping at Epoch: {epoch+1}  Accuracy {best_accuracy}, loss: {cost.item()}')
                    looper.close()
                    break
        looper.set_description(f'Epoch: {epoch+1}  Accuracy {accuracy}, loss: {cost.item()}')
    print(f'Best Accuracy {best_accuracy}')

    return best_accuracy, predicted_labels, decision_boundary, node


def get_index(dataloaders):
    # Takes the validation dataloader, if exists
    if len(dataloaders) > 1:
        return 1
    else:
        return 0


def get_decision_boundary(node):
    with torch.no_grad():
        w, b = [p for p in node.parameters()]
        return -b / w


def evaluate_accuracy(inputs, targets, node):
    predictions = node(inputs)
    labels = predictions > 0.5
    correctly_labelled = torch.sum(labels == targets)
    accuracy = 100. * correctly_labelled / len(targets)
    return accuracy, labels


def plot_perceptron(results, save_dir=None, show_plot=False, name='train'):
    fig = plt.figure()
    plt.title(f"Accuracy: {results['accuracy_value']:.2f} %")
    plt.plot(TorchUtils.get_numpy_from_tensor(results['norm_inputs']), label='Norm. Waveform')
    plt.plot(TorchUtils.get_numpy_from_tensor(results['predicted_labels']), '.', label='Predicted labels')
    plt.plot(TorchUtils.get_numpy_from_tensor(results['targets']), 'g', label='Targets')
    plt.plot(np.arange(len(results['predicted_labels'])),
             TorchUtils.get_numpy_from_tensor(torch.ones_like(results['predicted_labels']) * results['threshold']), 'k:', label='Threshold')
    plt.legend()
    if show_plot:
        plt.show()
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, name + '_accuracy.jpg'))
    plt.close()
    return fig
