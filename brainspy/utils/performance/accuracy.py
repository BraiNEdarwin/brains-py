import os
from xmlrpc.client import boolean
import torch
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt
from brainspy.utils.pytorch import TorchUtils
from brainspy.utils.performance.data import get_data


def get_accuracy(inputs, targets, configs=None, node=None):
    """
    To calculate the accuracy of the device on a binary classification task. Binary classification
    is the task of classifying the elements of a set into groups on the basis of a classification
    rule. This function helps finding a binary threshold for separating the output of a DNPU or
    DNPU architecture, and assigns a True/False label to the output depending on whether if the
    signal is above or below the discovered threshold. In this way, this function helps calculating
    the number of correctly classified binary targets out of the total number of targets, for a
    given dataset.

    To calculate the accuracy, a perceptron is trained using binary cross entropy on the output of
    the DNPU or DNPU architecture for the training dataset. The model trained consists of a single
    linear layer, where the sigmoid of the perceptron is included in the binary cross entropy loss
    function. For calculating the accuracy of the test and validation datasets, the already trained
    perceptron is passed to the function, and the method calculates the accuracy based on that.

    Refer to https://www.upgrad.com/blog/perceptron-learning-algorithm-how-it-works/ to see how a
    Perceptron works.

    The specific method for calculating the accuracy is:

        1. Normalises the input data (which is the output data of the DNPU or DNPU architecture)
        2. (Optional) Train a perceptron, only needed when using the normalised output of the DNPU
           or DNPU architecture corresponding to the training dataset of a particular task. To use
           it leave the option node=None.
        3. Pass the normalised output of the DNPU or DNPU architecture through the trained
           perceptron, and compare the output against the binary targets. This comparison is used
           to calculate the accuracy of the solution.
        4. Store all the data including results in a dictionary and return it

    Refer to:
    https://pytorch-lightning.readthedocs.io/en/1.2.6/_modules/pytorch_lightning/metrics/classification/accuracy.html
    to see how accuracy is calculated in PyTorch.

    Parameters
    ----------
    inputs : torch.Tensor
        The inputs to the perceptron algorithm, which are the outputs of the DNPU or DNPU
        architectures that you want to evaluate the accuracy against. Only input tensors
        with a single dimension are supported with the default node.

    targets : torch.Tensor
        Binary targets against which the outuut of the perceptron algorithm is compared.
        Only target tensors with a single dimension are supported with the default node.

    configs : dict, optional
        Configurations of the model to get the accuracy using the perceptron algorithm.
        The configs should contain a description of the hyperparameters that can be used for
        get_accuracy. By default is None. It has the following keys:

            epochs: int
                Number of loops used for training the perceptron. (default: 100)
            learning_rate: float
                Learning rate used to train the perceptron. (default: 1e-3)
            data:
                batch_size: int
                    Batch size used to train the perceptron. (default: 256)
                worker_no: int
                    How many subprocesses to use for data loading. 0 means that the data will be
                    loaded in the main process. (default: 0)
                pin_memory: boolean (default: False)
                    If True, the data loader will copy Tensors into CUDA pinned memory before
                    returning them.

    node : Optional[torch.nn.Module]
        Is the trained linear layer of the perceptron. Leave it as None if you want to train a
        perceptron from scratch. (default: None) The default perceptron only supports one
        dimensional outpus.

    Returns
    -------
    dict
        Results/accuracy of the algorithm in a dictionary with the following keys:

                    accuracy_value: float
                        Percentage accuracy obtained by calculating the number of
                        correct binary outputs against the targets.
                    node: torch.nn.Linear
                        The linear layer of the trained/used perceptron.
                    predicted_labels:
                        Predicted labels after passing the normalised inputs through
                        the perceptron.
                    norm_threshold : Normalised threshold used for the classification.
                    configs : Configurations of the node. It has the following keys:

                            epochs: int
                                Number of loops used for training the perceptron. (default: 100)
                            learning_rate: float
                                Learning rate used to train the perceptron. (default: 1e-3)
                            data:
                                batch_size: int
                                    Batch size used to train the perceptron. (default: 256)
                                worker_no: int
                                    How many subprocesses to use for data loading. 0 means that the
                                    data will be loaded in the main process. (default: 0)
                                pin_memory: boolean (default: False)
                                    If True, the data loader will copy Tensors into CUDA pinned
                                    memory before returning them.

    """
    assert type(inputs) == torch.Tensor and type(targets) == torch.Tensor
    if configs is not None:
        assert type(configs) == dict
    assert (len(inputs.shape) != 1
            and len(targets.shape) != 1), "Please unsqueeze inputs and targets"

    if configs is None:
        configs = get_default_node_configs()

    if node is None:
        train = True
        node = TorchUtils.format(torch.nn.Linear(1, 1))
    else:
        train = False

    results, dataloader = init_results(inputs, targets, configs)

    if train:
        optimizer = torch.optim.Adam(node.parameters(),
                                     lr=configs["learning_rate"])
        accuracy, node = train_perceptron(configs["epochs"],
                                          dataloader,
                                          optimizer=optimizer,
                                          node=node,
                                          stop_at_max_accuracy=True)

    with torch.no_grad():
        node.eval()
        accuracy, predicted_labels = evaluate_accuracy(results["norm_inputs"],
                                                       results["targets"],
                                                       node)
        w, b = [p for p in node.parameters()]
        threshold = -b / w

        results["norm_threshold"] = threshold.clone()
        results["threshold"] = (threshold *
                                inputs.std(dim=0)) + inputs.mean(dim=0)
        results["predicted_labels"] = predicted_labels
        results["node"] = node
        results["accuracy_value"] = accuracy
        results["configs"] = configs

    return results


def init_results(inputs, targets, configs):
    """
    To initialize the results of the accuracy test and the results of the Perceptron algorithm.
    The method initializes the results dictionary for evaluation of accuracy , and initializes the
    perceptron dataset from thge Dataloader (Refer to data.py for the Perceptron dataloader).

    Parameters
    ----------
    inputs : torch.Tensor
        The inputs to the perceptron algorithm, which are the outputs of the DNPU or DNPU
        architectures that you want to evaluate the accuracy against.

    targets : torch.Tensor
        Binary targets against which the output of the perceptron algorithm is compared.

    configs : dict, optional
        Configurations of the model to get the accuracy using the perceptron algorithm.
        The configs should contain a description of the hyperparameters that can be used for
        get_accuracy. By default is None. It has the following keys:

            epochs: int
                Number of loops used for training the perceptron. (default: 100)
            learning_rate: float
                Learning rate used to train the perceptron. (default: 1e-3)
            batch_size: int
                Batch size used to train the perceptron. (default: 256)

    Returns
    -------
    dict - initialized data for evaluation of accuracy
    torch.utils.data.Dataloader : results of the Perceptron dataloader
    """
    assert type(inputs) == torch.Tensor and type(targets) == torch.Tensor
    if configs is not None:
        assert type(configs) == dict
    results = {}
    results["inputs"] = inputs.clone()
    results["targets"] = targets.clone()
    results["norm_inputs"] = zscore_norm(inputs.clone())
    dataloader = get_data(results, configs["batch_size"])
    return results, dataloader


def zscore_norm(inputs, eps=1e-5):
    """
    To calculate the standard normal distribution from the input data.
    The standard normal distribution, represented by the letter Z, is the normal distribution
    having a mean of 0 and a standard deviation of 1.
    Refer to https://towardsdatascience.com/the-surprising-longevity-of-the-z-score-a8d4f65f64a0 to
    read about z-score and normalisation of data in PyTorch.

    Parameters
    ----------
    inputs : torch.Tensor
        The inputs to the perceptron algorithm, which are the outputs of the DNPU or DNPU
        architectures that you want to evaluate the accuracy against.
    eps : int , optional
        Value of epsilon. (default: 1e-5)

    Returns
    -------
    torch.Tensor
       Normalised inputs.
    """
    assert type(inputs) == torch.Tensor
    assert (
        inputs.std() != 0
    ), "The standard deviation of the inputs is 0. Please check that the inputs are correct. "

    return (inputs - inputs.mean(axis=0)) / inputs.std(dim=0)


def train_perceptron(
        epochs: int,
        dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.modules.loss._Loss = torch.nn.BCEWithLogitsLoss(),
        node: torch.nn.Module = None,
        stop_at_max_accuracy: bool = False):
    """
    To train the Perceptron obtain a set of weights w that accurately classifies each instance in
    our training set. In order to train our Perceptron, we iteratively feed the network with our
    training data multiple times.

    The perceptron is used as a linear classifier to facilitate supervised learning of binary
    classifiers. The objective of this learning problem is to use data with correct labels for
    making predictions for training a model. This supervised learning include classification to
    predict class labels.

    Refer to:
    https://medium.com/biaslyai/pytorch-introduction-to-neural-network-feedforward-neural-network-model-e7231cff47cb
    for an example of training a Perceptron in PyTorch.

    Parameters
    ----------
    epochs : int
        Number of epochs.

    dataloader : torch.utils.data.DataLoader
        The normalised output from the DNPU or DNPU architecture, already in a dataloader format.

    optimizer : torch.optim.Optimizer
        Optimization algorithm. (default: torch.optim.Adam)

    loss_fn : Optional[torch.nn.modules.loss._Loss]
       Loss functions are used to gauge the error between the prediction output and the provided
       target value, by default torch.nn.BCEWithLogitsLoss()

    node : Optional[torch.nn.Module]
        Is the trained linear layer of the perceptron. Leave it as None if you want to train a
        perceptron from scratch. (default: None)

    stop_at_max_accuracy : boolean
        Decides to immediately stop after achieving 100% solution if true. If false, it will keep 
        training to improve the threshold separation. Recommended to be true when doing many runs
        at the same time in tasks like capacity test or searcher.

    Returns
    -------
    accuracy : int - accuracy of the perceptron
    node : torch.nn -  node of the perceptron
    """
    assert type(epochs) == int
    looper = trange(epochs, desc="Calculating accuracy")
    node = node.to(device=TorchUtils.get_device(),
                   dtype=torch.get_default_dtype())

    for epoch in looper:
        evaluated_sample_no = 0
        running_loss = 0.
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
            correctly_labelled += int(
                torch.sum(labels == (targets == 1)).item())
            evaluated_sample_no += inputs.shape[0]

        accuracy = 100.0 * correctly_labelled / evaluated_sample_no
        running_loss /= evaluated_sample_no
        looper.set_description(
            f"Training perceptron: Epoch: {epoch}  Accuracy {accuracy}," +
            f" running loss: {running_loss}")
        if accuracy >= 100.0 and stop_at_max_accuracy:
            print("\nReached 100/% accuracy. Stopping.")
            break
    return accuracy, node


def evaluate_accuracy(inputs, targets, node):
    """
    To evaluate the accuracy of the Perceptron algorithm on the input data ( which is the outut of
    the DNPU or DNPU architecture ) based on the inputs and target values provided.
    The accuarcy is evaluated by passing the normalised output of the DNPU or DNPU architecture
    through the trained perceptron, and compare the output against the binary targets

    inputs : torch.Tensor
        The inputs to the perceptron algorithm, which are the outputs of the DNPU or DNPU
        architectures that you want to evaluate the accuracy against.

    targets : torch.Tensor
        Binary targets against which the output of the perceptron algorithm is compared.

    node : Optional[torch.nn.Module]
        Is the trained linear layer of the perceptron. Leave it as None if you want to train a
        perceptron from scratch. (default: None)

    Returns
    -------
    accuracy - int  - the accuracy calculated from the data provided
    labels - bool - if the predictions of the noda are greatrer than 0

    """
    assert type(inputs) == torch.Tensor and type(targets) == torch.Tensor
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
        configurations of the perceptron to calculate the accuracy on the input data (which is the
        output of the DNU or DNPU architecture). The dictionary contains the following keys:
            epochs : int
                Number of epochs.

            dataloader : torch.utils.data.Dataloader
                The normalised output from the DNPU or DNPU architecture, already in a dataloader
                format.

            optimizer : torch.optim.Optimizer
                Optimization algorithm. (default: torch.optim.Adam)

            loss_fn : Optional[torch.nn.modules.loss._Loss]
            Loss functions are used to gauge the error between the prediction output and the
            provided target value, by default torch.nn.BCEWithLogitsLoss()

            node : Optional[torch.nn.Module]
                Is the trained linear layer of the perceptron. Leave it as None if you want to
                train a perceptron from scratch. (default: None)
    """
    configs = {}
    configs["epochs"] = 100
    configs["learning_rate"] = 0.001
    configs["batch_size"] = 256
    return configs


def plot_perceptron(results, save_dir=None, show_plots=False, name="train"):
    """
    Plot the results of the perceptron algorithm. You can choose to see how the data has been
    plotted and also save the results to a specified directory.

    Parameters
    ----------
    results : dict
        Results/accuracy of the algorithm in a dictionary with the following keys:
                    accuracy_value : Percentage Accuracy of the Algorithm
                    node : the transformation applied to the dataset
                    predicted_labels : predicted labels used to calculate accuracy
                    norm_threshold : Threshold probability value for accuracy calculation of the
                    Perceptron
                    configs : configurations of the node

    save_dir : str, optional
        Directory in which you want to save the results. (default: None)

    show_plots: bool, optional
        To see how the perceptron plotted the data, by default False

    name : str, optional
        To train the data, by default "train".

    Returns
    -------
    matplotlib.pyplot.figure
        A new figure contaning the results.
    """
    assert type(results) == dict
    fig = plt.figure()
    plt.title(f"Accuracy: {results['accuracy_value']:.2f} %")
    plt.plot(TorchUtils.to_numpy(results["norm_inputs"]),
             ".",
             label="Norm. Waveform")
    plt.plot(
        TorchUtils.to_numpy(results["predicted_labels"]),
        ".",
        label="Predicted labels",
    )
    plt.plot(TorchUtils.to_numpy(results["targets"]), "g", label="Targets")
    plt.plot(
        np.arange(len(results["predicted_labels"])),
        TorchUtils.to_numpy(
            torch.ones_like(results["predicted_labels"]) *
            results["norm_threshold"]),
        "k:",
        label="Norm. Threshold",
    )
    plt.legend()
    if show_plots:
        plt.show()
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, name + "_accuracy.jpg"))
    plt.close(fig)
    return fig
