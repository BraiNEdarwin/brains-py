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
import warnings
from brainspy.algorithms.modules.performance.data import get_data

from brainspy.utils.pytorch import TorchUtils


def get_accuracy(inputs, targets, configs=None, node=None):
    # Assumes that the input_waveform and the target_waveform have the shape (n_total,1)
    # Normalizes the data; it is assumed that the target_waveform has binary values

    assert (len(inputs.shape) != 1
            and len(targets.shape) != 1), "Please unsqueeze inputs and targets"

    # Initialise node configs
    if configs is None:
        configs = get_default_node_configs()

    # Initialise perceptron
    if node is None:
        train = True
        node = torch.nn.Linear(1, 1)
    else:
        train = False

    # Initialise results dictionary
    results, dataloader = init_results(inputs, targets, configs)

    if train:
        optimizer = torch.optim.Adam(node.parameters(),
                                     lr=configs["learning_rate"])
        accuracy, node = train_perceptron(configs['epochs'],
                                          dataloader,
                                          optimizer,
                                          node=node)

    with torch.no_grad():
        node.eval()
        accuracy, predicted_labels = evaluate_accuracy(results['norm_inputs'],
                                                       results['targets'],
                                                       node)
        w, b = [p for p in node.parameters()]
        threshold = -b / w

        # Save remaining results dictionary
        results["norm_threshold"] = threshold.clone()
        results["threshold"] = (threshold * inputs.std(dim=0)) + inputs.mean(dim=0)
        results["predicted_labels"] = predicted_labels
        results["node"] = node
        results["accuracy_value"] = accuracy
        results['configs'] = configs

    return results


def init_results(inputs, targets, configs):
    results = {}
    results["inputs"] = inputs.clone()
    results["targets"] = targets.clone()
    results["norm_inputs"] = zscore_norm(inputs.clone())
    dataloader = get_data(results, configs)
    return results, dataloader


def zscore_norm(inputs, eps=1e-5):
    assert inputs.std(
        dim=0
    ) != 0, "The standard deviation of the inputs is 0. Please check that the inputs are correct. "

    return (inputs - inputs.mean(axis=0)) / inputs.std(dim=0)


def train_perceptron(epochs,
                     dataloader,
                     optimizer,
                     loss_fn=torch.nn.BCEWithLogitsLoss(),
                     node=None):

    looper = trange(epochs, desc="Calculating accuracy")
    # looper = tqdm(epochs)
    node = node.to(device=TorchUtils.get_device(),
                   dtype=torch.get_default_dtype())
    # validation_index = get_index(dataloaders)

    for epoch in looper:
        evaluated_sample_no = 0
        running_loss = 0
        correctly_labelled = 0
        for inputs, targets in dataloader:
            if inputs.device != TorchUtils.get_device():
                inputs = inputs.to(TorchUtils.get_device())
            if targets.device != TorchUtils.get_device():
                targets = targets.to(TorchUtils.get_device())
            optimizer.zero_grad()
            predictions = node(inputs)
            loss = loss_fn(predictions, targets)
            loss.backward()
            optimizer.step()
            running_loss = loss.item() * (inputs.shape[0])
            labels = predictions > 0.
            correctly_labelled += torch.sum(labels == (targets == 1.))
            evaluated_sample_no += inputs.shape[0]       
        #     f"Epoch: {epoch}  Accuracy {correctly_labelled/evaluated_sample_no}, running loss: {running_loss/evaluated_sample_no}"
        # )
        accuracy = 100.0  * correctly_labelled / evaluated_sample_no
        running_loss /= evaluated_sample_no
        looper.set_description(f"Training perceptron: Epoch: {epoch}  Accuracy {accuracy}, running loss: {running_loss}")
        # print(
        #     f"Epoch: {epoch}  Accuracy {accuracy}, running loss: {running_loss}"
        # )
        if accuracy >= 100.0:
            print(f"Reached 100/% accuracy. Stopping.")
            #looper.close()
            break
    return accuracy, node


def evaluate_accuracy(inputs, targets, node):
    w, b = [p for p in node.parameters()]
    threshold = -b / w
    predictions = node(inputs)
    labels = predictions > 0.
    correctly_labelled = torch.sum(labels == (targets == 1.))
    accuracy = 100.0 * correctly_labelled / len(targets)
    return accuracy, labels


def get_default_node_configs():
    configs = {}
    configs["epochs"] = 100
    configs["learning_rate"] = 0.001
    configs["data"] = {}
    configs["data"]["batch_size"] = 256
    configs["data"]["worker_no"] = 0
    configs["data"]["pin_memory"] = False
    return configs


def plot_perceptron(results, save_dir=None, show_plot=False, name="train"):
    fig = plt.figure()
    plt.title(f"Accuracy: {results['accuracy_value']:.2f} %")
    plt.plot(TorchUtils.to_numpy(results["norm_inputs"]),
             '.',
             label="Norm. Waveform")
    plt.plot(
        TorchUtils.to_numpy(results["predicted_labels"]),
        ".",
        label="Predicted labels",
    )
    plt.plot(TorchUtils.to_numpy(results["targets"]),
             "g",
             label="Targets")
    plt.plot(
        np.arange(len(results["predicted_labels"])),
        TorchUtils.to_numpy(
            torch.ones_like(results["predicted_labels"]) *
            results["norm_threshold"]),
        "k:",
        label="Norm. Threshold",
    )
    plt.legend()
    if show_plot:
        plt.show()
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, name + "_accuracy.jpg"))
    plt.close()
    return fig
