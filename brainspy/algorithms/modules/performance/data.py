import torch
from torch.utils.data import SubsetRandomSampler, Dataset, random_split

from brainspy.utils.pytorch import TorchUtils


def get_data(results, configs):
    """
    Get the Perceptron data.
    This method prepares the dataset for the perceptron algorithm based on the configurations of the model
    and required rseults. The dataset is then returned as a torch Dataloader which can be given to the Perceptron algorithm.

    Parameters
    ----------
    results : torch.Tensor
         target values required for this perceptron algorithm
    configs : dict
        configurations of the model

    Returns
    -------
    torch.utils.data.Dataloader
        Dataloader of the perceptron algorithm
    """
    if configs["data"]["worker_no"] > 0 or configs["data"]["pin_memory"]:
        dataset = PerceptronDataset(
            results["norm_inputs"], results["targets"], device=torch.device("cpu")
        )
    else:
        dataset = PerceptronDataset(results["norm_inputs"], results["targets"])
    dataloaders = torch.utils.data.DataLoader(
        dataset,
        batch_size=configs["data"]["batch_size"],
        shuffle=True,
        num_workers=configs["data"]["worker_no"],
        pin_memory=configs["data"]["pin_memory"],
    )
    return dataloaders


class PerceptronDataset(Dataset):
    """
    Class which creates a dataset for the Perceptron algorithm
    The Perceptron is a linear machine learning algorithm for binary classification tasks.
    ike logistic regression, it can quickly learn a linear separation in feature space for two-class classification tasks,
    although unlike logistic regression, it learns using the stochastic gradient descent optimization algorithm and does not predict calibrated probabilities.
    """

    def __init__(self, inputs, targets, device=None):
        """Initialize the dataset of the Perceptron

        Parameters
        ----------
        inputs : torch.Tensor
            The Perceptron receives multiple input signals,it either outputs a signal or does not return an output.
        targets : torch.Tensor
            target values required for this perceptron algorithm
        device : torch.Device, optional
            torch device is CUDA or CPU, by default None
        """

        # Normalise inputs
        assert len(inputs) > 10, "Not enough data, at least 10 points are required."
        assert not torch.isnan(inputs).any(), "NaN values detected."
        if device is None:
            self.inputs = inputs.to(dtype=torch.get_default_dtype())
            self.targets = targets.to(dtype=torch.get_default_dtype())
        else:
            self.inputs = inputs.to(device=device, dtype=torch.get_default_dtype())
            self.targets = targets.to(device=device, dtype=torch.get_default_dtype())

    def __getitem__(self, index):
        """
        Gets the input and target at a given index in this Perceptron dataset

        Parameters
        ----------
        index : int
            position/index of the required input and target

        Returns
        -------
        (int,int)
            tuple of input and target at a given index
        """
        inputs = self.inputs[index, :]
        targets = self.targets[index, :]
        return (inputs, targets)

    def __len__(self):
        """Get the lengh=th of the input values

        Returns
        -------
        int
            length of the input dataset
        """
        return len(self.inputs)
