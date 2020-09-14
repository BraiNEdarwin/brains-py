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

from brainspy.algorithms.modules.performance.data import get_data

from brainspy.utils.pytorch import TorchUtils


def get_accuracy(inputs, targets, configs=None, node=None):
    # Assumes that the input_waveform and the target_waveform have the shape (n_total,1)
    # Normalizes the data; it is assumed that the target_waveform has binary values

    assert (
        len(inputs.shape) != 1 and len(targets.shape) != 1
    ), "Please unsqueeze inputs and targets"

    if configs is None:
        configs = get_default_node_configs()
    # Initialise perceptron
    if node is None:
        train = True
        node = torch.nn.Linear(1, 1)
    else:
        train = False

    # Initialise results dictionary
    results = {}
    results["inputs"] = inputs.clone()
    std = inputs.std(axis=0)
    if std == 0:  # This is to avoid nan values when normalising the input.
        std = 1
    results["norm_inputs"] = (inputs - inputs.mean(axis=0)) / std
    results["targets"] = targets

    if train:
        train_dataloaders = get_data(results, configs, shuffle=True)
        accuracy, predicted_labels, node = train_perceptron(
            train_dataloaders, configs, node
        )
    # else:
    #     # TODO: Support validation on this modality
    node.eval()
    dataloaders = get_data(results, configs, shuffle=False)

    accuracy, predicted_labels = evaluate_accuracy(
        dataloaders[0], node
    )
    threshold = get_decision_boundary(node)
    #threshold = get_decision_boundary(node)
    print("Best accuracy: " + str(accuracy))

    # Save remaining results dictionary
    # results['predictions'] = predictions
    results["norm_threshold"] = threshold.clone()
    results["threshold"] = threshold * inputs.std(dim=0) + inputs.mean(dim=0)
    results["predicted_labels"] = predicted_labels
    results["node"] = node
    results["accuracy_value"] = accuracy
    results['configs'] = configs

    return results


def get_default_node_configs():
    configs = {}
    configs["epochs"] = 100
    configs["learning_rate"] = 0.0007
    configs["betas"] = [0.999, 0.999]
    configs["split"] = [1, 0]
    configs["mini_batch"] = 256
    return configs


def train_perceptron(dataloaders, configs, node=None):
    # Initialise key elements of the trainer

    loss = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        node.parameters(), lr=configs["learning_rate"], betas=configs["betas"]
    )
    best_accuracy = -1
    best_labels = None
    looper = trange(configs["epochs"], desc="Calculating accuracy")
    node = node.to(device=TorchUtils.get_accelerator_type(), dtype=torch.float16)
    validation_index = get_index(dataloaders)

    for epoch in looper:
        for inputs, targets in dataloaders[0]:
            if inputs.device != TorchUtils.get_accelerator_type():
                inputs = inputs.to(TorchUtils.get_accelerator_type())
            if targets.device != TorchUtils.get_accelerator_type():
                targets = targets.to(TorchUtils.get_accelerator_type())
            optimizer.zero_grad()
            predictions = node(inputs)
            cost = loss(predictions, targets)
            cost.backward()
            optimizer.step()
        with torch.no_grad():
            node.eval()
            accuracy, labels = evaluate_accuracy(dataloaders[validation_index], node)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_labels = labels
                #decision_boundary = get_decision_boundary(node)
                # TODO: Add a more efficient stopping mechanism ?
                if best_accuracy >= 100.0:
                    looper.set_description(
                        f"Reached 100/% accuracy. Stopping at Epoch: {epoch+1}  Accuracy {best_accuracy}, loss: {cost}"
                    )
                    looper.close()
                    break
            node.train()
        looper.set_description(
            f"Epoch: {epoch+1}  Accuracy {accuracy}, loss: {cost}"
        )
    print(f"Best perceptron accuracy during perceptron training: {best_accuracy}")

    return best_accuracy, best_labels, node


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


def evaluate_accuracy(dataloader, node):
    correctly_labelled = 0
    i = 0
    labels = torch.zeros(len(dataloader.dataset), device=TorchUtils.get_accelerator_type(), dtype=torch.bool)

    for inputs, targets in dataloader:
        if inputs.device != TorchUtils.get_accelerator_type():
            inputs = inputs.to(TorchUtils.get_accelerator_type())
        if targets.device != TorchUtils.get_accelerator_type():
            targets = targets.to(TorchUtils.get_accelerator_type())
        # accuracy, predicted_labels = evaluate_accuracy(inputs, targets, node)
        predictions = node(inputs)
        labels[i:i + len(predictions)] = (predictions > 0.).squeeze(dim=1)
        correctly_labelled += torch.sum(labels[i:i + len(predictions)] == targets.squeeze(dim=1))

        i += dataloader.batch_size
    accuracy = 100.0 * correctly_labelled / len(dataloader.dataset)
    return accuracy, labels.unsqueeze(dim=1)  # , inputs, targets


def plot_perceptron(results, save_dir=None, show_plot=False, name="train"):
    fig = plt.figure()
    plt.title(f"Accuracy: {results['accuracy_value']:.2f} %")
    plt.plot(
        TorchUtils.get_numpy_from_tensor(results["norm_inputs"]), label="Norm. Waveform"
    )
    plt.plot(
        TorchUtils.get_numpy_from_tensor(results["predicted_labels"]),
        ".",
        label="Predicted labels",
    )
    plt.plot(TorchUtils.get_numpy_from_tensor(results["targets"]), "g", label="Targets")
    plt.plot(
        np.arange(len(results["predicted_labels"])),
        TorchUtils.get_numpy_from_tensor(
            torch.ones_like(results["predicted_labels"]) * results["norm_threshold"]
        ),
        "k:",
        label="Threshold",
    )
    plt.legend()
    if show_plot:
        plt.show()
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, name + "_accuracy.jpg"))
    plt.close()
    return fig
