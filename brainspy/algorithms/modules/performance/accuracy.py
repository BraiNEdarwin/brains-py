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

from brainspy.algorithms.modules.performance.data import get_data, DTYPE

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
    results = init_results(inputs, targets)

    if train:
        accuracy, predicted_labels, node = train_perceptron(
            results, configs, node
        )

    with torch.no_grad():
        node.eval()
        accuracy, predicted_labels = evaluate_accuracy(
            results['norm_inputs'], results['targets'], node
        )
        w, b = [p for p in node.parameters()]
        threshold = -b / w

        # Save remaining results dictionary
        results["norm_threshold"] = threshold.clone()
        results["threshold"] = threshold * inputs.std(dim=0) + inputs.mean(dim=0)
        results["predicted_labels"] = predicted_labels
        results["node"] = node
        results["accuracy_value"] = accuracy
        results['configs'] = configs

        print("Best accuracy: " + str(accuracy))

    return results


def init_results(inputs, targets, eps=1e-5):
    results = {}
    results["inputs"] = inputs.clone()
    std = inputs.std(axis=0)
    if std == 0:  # This is to avoid nan values when normalising the input.
        std = eps
    results["norm_inputs"] = (inputs - inputs.mean(axis=0)) / std
    results["targets"] = targets
    return results


def train_perceptron(results, configs, node=None):
    # Initialise key elements of the trainer
    dataloaders = get_data(results, configs, shuffle=True)
    loss = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        node.parameters(), lr=configs["learning_rate"], betas=configs["betas"]
    )
    best_accuracy = -1
    best_labels = None
    looper = trange(configs["epochs"], desc="Calculating accuracy")
    node = node.to(device=TorchUtils.get_accelerator_type(), dtype=DTYPE)
    # validation_index = get_index(dataloaders)

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
            accuracy, labels = evaluate_accuracy(results['norm_inputs'], results['targets'], node)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_labels = labels
                w, b = [p for p in node.parameters()]
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
            #f"Epoch: {epoch+1} loss: {cost}"
        )
    node.weight = w
    node.bias = b
    return best_accuracy, best_labels, node


def evaluate_accuracy(inputs, targets, node):
    predictions = node(inputs)
    labels = predictions > 0.
    correctly_labelled = torch.sum(labels == targets)
    accuracy = 100.0 * correctly_labelled / len(targets)
    return accuracy, labels


def get_default_node_configs():
    configs = {}
    configs["epochs"] = 100
    configs["learning_rate"] = 0.0007
    configs["betas"] = [0.999, 0.999]
    configs["data"] = {}
    configs["data"]["split"] = [1, 0]
    configs["data"]["mini_batch"] = 256
    configs["data"]["worker_no"] = 0
    configs["data"]["pin_memory"] = False
    return configs


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
        label="Norm. Threshold",
    )
    plt.legend()
    if show_plot:
        plt.show()
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, name + "_accuracy.jpg"))
    plt.close()
    return fig
