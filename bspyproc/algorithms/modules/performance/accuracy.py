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

from bspyalgo.algorithms.modules.performance.perceptron import Perceptron, PerceptronDataset
from bspyproc.utils.pytorch import TorchUtils


def get_accuracy(inputs, targets, split=[1, 0], node=None):
    # Assumes that the input_waveform and the target_waveform have the shape (n_total,1)
    # Normalizes the data; it is assumed that the target_waveform has binary values

    assert len(inputs.shape) != 1 and len(targets.shape) != 1, "Please unsqueeze inputs and targets"

    # Initialise perceptron
    if node is None:
        train = True
        node = Perceptron()
    else:
        train = False

    # Initialise results dictionary
    results = {}
    results['inputs'] = inputs.clone()
    results['norm_inputs'] = (inputs - torch.mean(inputs, axis=0)) / torch.std(inputs, axis=0)
    results['targets'] = targets

    if train:
        # Prepare perceptron data
        dataset = PerceptronDataset(results['norm_inputs'], results['targets'])
        dataloaders = random_split(dataset, split)
        if len(dataloaders[1]) == 0:
            dataloaders[1] = dataloaders[0]
        # Train the perceptron
        accuracy, predictions, threshold, node = train_perceptron(dataloaders, node)
        print('Best accuracy: ' + str(accuracy.item()))
    else:
        accuracy, predicted_class = evaluate_accuracy(results['norm_inputs'], results['targets'], node)
        threshold = get_decision_boundary(node)
        print('Best accuracy: ' + str(accuracy.item()))

    # Save remaining results dictionary
    results['predictions'] = predictions
    results['threshold'] = threshold
    results['predicted_class'] = predicted_class
    results['node'] = node
    results['accuracy_value'] = accuracy

    return results


def train_perceptron(dataloaders, node=None, lrn_rate=0.0007, mini_batch=8, epochs=100, validation=False, verbose=True):
    # Initialise key elements of the trainer
    node = TorchUtils.format_tensor(node)
    loss = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(node.parameters(), lr=lrn_rate, betas=(0.999, 0.999))
    best_accuracy = -1
    looper = trange(epochs, desc='Calculating accuracy')

    for epoch in looper:
        for inputs, targets in dataloaders[0]:
            optimizer.zero_grad()
            predictions = node(inputs)
            cost = loss(predictions, targets)
            cost.backward()
            optimizer.step()
        with torch.no_grad():
            inputs, targets = dataloaders[1].dataset[:]
            accuracy, predicted_class = evaluate_accuracy(inputs, targets, node)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                decision_boundary = get_decision_boundary(node)
                if best_accuracy >= 100.:
                    looper.set_description(f'Reached 100/% accuracy. Stopping at Epoch: {epoch+1}  Accuracy {best_accuracy}, loss: {cost.item()}')
                    break
        if verbose:
            looper.set_description(f'Epoch: {epoch+1}  Accuracy {best_accuracy}, loss: {cost.item()}')

    return best_accuracy, predicted_class, decision_boundary, node


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
    plt.plot(TorchUtils.get_numpy_from_tensor(results['predictions']), '.', label='Predicted labels')
    plt.plot(TorchUtils.get_numpy_from_tensor(results['targets']), 'g', label='Targets')
    plt.plot(np.arange(len(results['predictions'])),
             TorchUtils.get_numpy_from_tensor(torch.ones_like(results['predictions']) * results['threshold']), 'k:', label='Threshold')
    plt.legend()
    if show_plot:
        plt.show()
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, name + '_accuracy.jpg'))
    plt.close()
    return fig
