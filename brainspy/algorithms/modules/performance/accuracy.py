#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 11:42:27 2018

@author: hruiz and ualegre
"""

import os
import torch
import warnings
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt
from brainspy.utils.pytorch import TorchUtils
from brainspy.algorithms.modules.performance.data import get_data


def get_accuracy(inputs, targets, configs=None, node=None):
    """
    To calculate the accuracy of the device on a binary task.

    Binary classification is the task of classifying the elements of a set into groups on the basis of a classification rule. The following classifiers can be used :

    Boolean classifier:
        A single boolean gate classifier
    Ring classifier:
        A particular ring classifier with a given separation gap
        Multiple runs on a ring classifier with a given separation gap

    This classification rule can be defined in the configurations dict. The configs contain the hyperparameters to get the accuracy of the perceptron.

    To calculate the accuracy, a perceptron is trained using binary cross entropy on the output of the DNPU or DNPU architecture for the training dataset.
    The trained perceptron is applied to the test and validation test. The method evaluates the Perceptron algorithm on the synthetic dataset and reports the average accuracy across the n-fold cross-validation.
    It is assumed that the input_waveform and the target_waveform have the shape (n_total,1) and that the target_waveform has binary values.

    The Perceptron is a linear machine learning algorithm for binary classification tasks.
    Like logistic regression, it can quickly learn a linear separation in feature space for two-class classification tasks,
    although unlike logistic regression, it learns using the stochastic gradient descent optimization algorithm and does not predict calibrated probabilities.

    Refer to https://www.upgrad.com/blog/perceptron-learning-algorithm-how-it-works/ to see how a Perceptron works

    The method :

        1. Normalises the input data (which is the output data of the DNPU or DNPU architecture)
        2. (Optional) Train a perceptron, only needed when using the normalised output of the DNPU or DNPU architecture corresponding to the training dataset of a particular task. To use it leave the option node=None.
        3. Pass the normalised output of the DNPU or DNPU architecture through the trained perceptron, and compare the output against the binary targets. This comparison is used to calculate the accuracy of the solution.
        4. Store all the data including results in a dictionary and return it

    Refer to https://pytorch-lightning.readthedocs.io/en/1.2.6/_modules/pytorch_lightning/metrics/classification/accuracy.html to see how accuracy is calculated in PyTorch.

    Parameters
    ----------
    inputs : torch.Tensor
        the inputs to the perceptron algorithm, which are the outputs of the DNPU or DNPU architectures that you want to evaluate the accuracy against
    targets : torch.Tensor
        binary targets against which the outuut of the perceptron algorithm is compared
    configs : dict, optional
        configurations of the model to get the accuracy using the perceptron algorithm.
        The configs should contain a description of the hyperparameters that can be used for get_accuracy., by default None
    node : torch.nn., optional
        Is the trained perceptron. Leave it as None if you want to train a perceptron from scratch, by default None

    Returns
    -------
    dict
        Results/accuracy of the algorithm in a dictionary with the following keys:

                    accuracy_value : Percentage Accuracy of the Algorithm
                    node : the transformation applied to the dataset
                    predicted_labels : predicted labels used to calculate accuracy
                    norm_threshold : Threshold probability value for accuracy calculation of the Perceptron
                    configs : configurations of the node

    """

    assert (
        len(inputs.shape) != 1 and len(targets.shape) != 1
    ), "Please unsqueeze inputs and targets"

    if configs is None:
        configs = get_default_node_configs()

    if node is None:
        train = True
        node = torch.nn.Linear(1, 1)
    else:
        train = False

    results, dataloader = init_results(inputs, targets, configs)

    if train:
        optimizer = torch.optim.Adam(node.parameters(), lr=configs["learning_rate"])
        accuracy, node = train_perceptron(
            configs["epochs"], dataloader, optimizer, node=node
        )

    with torch.no_grad():
        node.eval()
        accuracy, predicted_labels = evaluate_accuracy(
            results["norm_inputs"], results["targets"], node
        )
        w, b = [p for p in node.parameters()]
        threshold = -b / w

        results["norm_threshold"] = threshold.clone()
        results["threshold"] = (threshold * inputs.std(dim=0)) + inputs.mean(dim=0)
        results["predicted_labels"] = predicted_labels
        results["node"] = node
        results["accuracy_value"] = accuracy
        results["configs"] = configs

    return results


def init_results(inputs, targets, configs):
    """
    To initialize the results of the accuracy test and the results of the Perceptron algorithm.
    The method initializes the results dictionary for evaluation of accuracy , and initializes the perceptron dataset from thge Dataloader
    (Refer to data.py for the Perceptron dataloader)

    Parameters
    ----------
    inputs : torch.Tensor
        the inputs to the perceptron algorithm, which are the outputs of the DNPU or DNPU architectures that you want to evaluate the accuracy against
    targets : torch.Tensor
        binary targets against which the outuut of the perceptron algorithm is compared
    configs : dict, optional
        configurations of the model to get the accuracy using the perceptron algorithm.
        The configs should contain a description of the hyperparameters that can be used for get_accuracy., by default None

    Returns
    -------
    dict - initialized data for evaluation of accuracy
    torch.utils.data.Dataloader : results of the Perceptron dataloader
    """
    results = {}
    results["inputs"] = inputs.clone()
    results["targets"] = targets.clone()
    results["norm_inputs"] = zscore_norm(inputs.clone())
    dataloader = get_data(results, configs)
    return results, dataloader


def zscore_norm(inputs, eps=1e-5):
    """
    To calculate the standard normal distribution from the input data.
    The standard normal distribution, represented by the letter Z, is the normal distribution having a mean of 0 and a standard deviation of 1.
    Refer to https://towardsdatascience.com/the-surprising-longevity-of-the-z-score-a8d4f65f64a0 to read about zcore and normalisation of data in PyTorch.

    Parameters
    ----------
    inputs : torch.Tensor
        the inputs to the perceptron algorithm, which are the outputs of the DNPU or DNPU architectures that you want to evaluate the accuracy against
    eps : int , optional
        value of epsilon , by default 1e-5

    Returns
    -------
    torch.Tensor
       normalized distibution of data
    """
    assert (
        inputs.std() != 0
    ), "The standard deviation of the inputs is 0. Please check that the inputs are correct. "

    return (inputs - inputs.mean(axis=0)) / inputs.std(dim=0)


def train_perceptron(
    epochs, dataloader, optimizer, loss_fn=torch.nn.BCEWithLogitsLoss(), node=None
):
    """
    To train the Perceptron obtain a set of weights w that accurately classifies each instance in our training set.
    In order to train our Perceptron, we iteratively feed the network with our training data multiple times.

    The perceptron is used as a linear classifier to facilitate supervised learning of binary classifiers.
    The objective of this learning problem is to use data with correct labels for making predictions for training a model.
    This supervised learning include classification to predict class labels.

    Refer to https://medium.com/biaslyai/pytorch-introduction-to-neural-network-feedforward-neural-network-model-e7231cff47cb for an example of training a Perceptron in PyTorch.

    Parameters
    ----------
    epochs : int
        number of epochs - The data can be shuffled. After each epoch the data is shuffled automatically.
    dataloader : torch.utils.data.Dataloader
        results of the Perceptron dataloader
    optimizer : torch.optim
        Optimization algorithm
    loss_fn : torch.nn, optional
       Loss functions are used to gauge the error between the prediction output and the provided target value, by default torch.nn.BCEWithLogitsLoss()
    node : torch.nn., optional
        Is the trained perceptron. Leave it as None if you want to train a perceptron from scratch, by default None


    Returns
    -------
    accuracy : int - accuracy of the perceptron
    node : torch.nn -  node of the perceptron
    """
    looper = trange(epochs, desc="Calculating accuracy")
    node = node.to(device=TorchUtils.get_device(), dtype=torch.get_default_dtype())

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
            labels = predictions > 0.0
            correctly_labelled += torch.sum(labels == (targets == 1.0))
            evaluated_sample_no += inputs.shape[0]

        accuracy = 100.0 * correctly_labelled / evaluated_sample_no
        running_loss /= evaluated_sample_no
        looper.set_description(
            f"Training perceptron: Epoch: {epoch}  Accuracy {accuracy}, running loss: {running_loss}"
        )
        if accuracy >= 100.0:
            print(f"Reached 100/% accuracy. Stopping.")
            break
    return accuracy, node


def evaluate_accuracy(inputs, targets, node):
    """
    To evaluate the accuracy of the Perceptron algorithm on the input data ( which is the outut of the DNPU or DNPU architecture ) based on the inputs and target values provided.
    The accuarcy is evaluated by passing the normalised output of the DNPU or DNPU architecture through the trained perceptron, and compare the output against the binary targets

    inputs : torch.Tensor
        the inputs to the perceptron algorithm, which are the outputs of the DNPU or DNPU architectures that you want to evaluate the accuracy against
    targets : torch.Tensor
        binary targets against which the outuut of the perceptron algorithm is compared
    node : torch.nn., optional
        Is the trained perceptron. It applies the neccesary transformation to the incoming data

    Returns
    -------
    accuracy - int  - the accuracy calculated from the data provided
    labels - bool - if the predictions of the noda are greatrer than 0

    """
    w, b = [p for p in node.parameters()]
    threshold = -b / w
    predictions = node(inputs)
    labels = predictions > 0.0
    correctly_labelled = torch.sum(labels == (targets == 1.0))
    accuracy = 100.0 * correctly_labelled / len(targets)
    return accuracy, labels


def get_default_node_configs():
    """
    To get a default configuration of the node of a perceptron.
    This method is used in the get_accuracy method if a node is not provided.

    Returns
    -------
    dict
        configurations of the perceptron to calculate the accuracy on the input data (which is the output of the DNU or DNPU architecture)
    """
    configs = {}
    configs["epochs"] = 100
    configs["learning_rate"] = 0.001
    configs["data"] = {}
    configs["data"]["batch_size"] = 256
    configs["data"]["worker_no"] = 0
    configs["data"]["pin_memory"] = False
    return configs


def plot_perceptron(results, save_dir=None, show_plot=False, name="train"):
    """
    Plot the results of the perceptron algorithm.
    You can choose to see how the data has been plotted and also save the results to a specified directory.

    Parameters
    ----------
    results : dict
        Results/accuracy of the algorithm in a dictionary with the following keys:

                    accuracy_value : Percentage Accuracy of the Algorithm
                    node : the transformation applied to the dataset
                    predicted_labels : predicted labels used to calculate accuracy
                    norm_threshold : Threshold probability value for accuracy calculation of the Perceptron
                    configs : configurations of the node

    save_dir : str, optional
        directory in wwhich you want to save the results, by default None
    show_plot : bool, optional
        To see how the perceptron plotted the data, by default False
    name : str, optional
        to train the data, by default "train"

    Returns
    -------
    matplotlib.pyplot.figure
        a new figure contaning the results
    """
    fig = plt.figure()
    plt.title(f"Accuracy: {results['accuracy_value']:.2f} %")
    plt.plot(TorchUtils.to_numpy(results["norm_inputs"]), ".", label="Norm. Waveform")
    plt.plot(
        TorchUtils.to_numpy(results["predicted_labels"]),
        ".",
        label="Predicted labels",
    )
    plt.plot(TorchUtils.to_numpy(results["targets"]), "g", label="Targets")
    plt.plot(
        np.arange(len(results["predicted_labels"])),
        TorchUtils.to_numpy(
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
