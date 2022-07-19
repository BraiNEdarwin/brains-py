import torch
from torch.utils.data import Dataset


def get_data(results, batch_size):
    """
    Initialises the perceptron Dataset and loads the dataset into the Pytorch Dataloader.
    The dataloader loads the data into the memory according to the batch size.
    Refer to https://pytorch.org/tutorials/beginner/basics/data_tutorial.html for DataLoaders
    in PyTorch.

    The data can be shuffled. After each epoch the data is shuffled automatically. This is by
    design to accelerate and improve the model training process.Because of this, the learning
    algorithm is stochastic and may achieve different results each time it is run.
    Refer to https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html to see how epochs
    are used when training a Classifier.

    Parameters
    ----------
    results : dict
        These contain the input and target values of the perceptron alogorithm.
        It also contains the normalised input data from which the Pytorch dataloader is created.

        It has the following keys:

        inputs : torch.Tensor
            The inputs to the perceptron algorithm, which are the outputs of the DNPU or DNPU
            architectures that you want to evaluate the accuracy against.
        norm_inputs : torch.Tensor
            Standard normal distribution of the input data. To calculate this, the
            zscore_norm function can be used in brainspy.utils.performance.accuracy
        targets : torch.Tensor
            Binary targets against which the outuut of the perceptron algorithm is compared.

    batch size- The batch size defines the number of samples that will be propagated
                            through the network.

    Returns
    -------
    torch.utils.data.Dataloader
        Dataloader of the perceptron algorithm
    """

    assert type(results) == dict, "Results field should be of type - dict"
    assert type(batch_size) == int, "Batch size should be of type - int"
    assert batch_size > 0, "batch_size should be a positive integer value"

    assert type(
        results["inputs"]
    ) == torch.Tensor, "Input data should be of type - torch.Tensor"
    assert type(
        results["norm_inputs"]
    ) == torch.Tensor, "Normalized Input data should be of type - torch.Tensor"
    assert type(
        results["targets"]
    ) == torch.Tensor, "Target data should be of type - torch.Tensor"

    dataset = PerceptronDataset(results["norm_inputs"], results["targets"])
    dataloaders = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    return dataloaders


class PerceptronDataset(Dataset):
    """
    This class is an instace of the Pytorch Dataset. It passes all the information onto the Pytorch
    dataset. The dataset stores the samples and their corresponding labels, and DataLoader wraps an
    iterable around the Dataset to enable easy access to the samples.

    Refer to https://pytorch.org/tutorials/beginner/basics/data_tutorial.html to see how Pytorch
    datasets are created and used.

    """
    def __init__(self, inputs, targets, device=None):
        """
        Initialize the dataset of the Perceptron

        Parameters
        ----------
        inputs : torch.Tensor
            the inputs to the perceptron algorithm, which are the outputs of the DNPU or DNPU
            architectures that you want to evaluate the accuracy against
        targets : torch.Tensor
            binary targets against which the outuut of the perceptron algorithm is compared
        device : torch.Device, optional
            torch device is CUDA or CPU, by default None
        """

        # Normalise inputs
        assert len(
            inputs) > 10, "Not enough data, at least 10 points are required."
        assert not torch.isnan(inputs).any(), "NaN values detected."
        if device is None:
            self.inputs = inputs.to(dtype=torch.get_default_dtype())
            self.targets = targets.to(dtype=torch.get_default_dtype())
        else:
            self.inputs = inputs.to(device=device,
                                    dtype=torch.get_default_dtype())
            self.targets = targets.to(device=device,
                                      dtype=torch.get_default_dtype())

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
        """
        Get the length of the input values

        Returns
        -------
        int
            length of the input dataset
        """
        return len(self.inputs)
