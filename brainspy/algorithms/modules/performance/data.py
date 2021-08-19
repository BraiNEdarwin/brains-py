import torch
from torch.utils.data import Dataset


def get_data(results, configs):
    """
    Initialises the perceptron Dataset and loads the dataset into the Pytorch Dataloader.
    The dataloader loads the data into the memory according to the configs dictionary.
    Refer to https://pytorch.org/tutorials/beginner/basics/data_tutorial.html for DataLoaders
    in PyTorch.

    The data can be shuffled. After each epoch the data is shuffled automatically. This is by
    design to accelerate and improve the model training process.Because of this, the learning
    algorithm is stochastic and may achieve different results each time it is run.
    Refer to https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html to see how epochs
    are used when training a Classifier.

    Parameters
    ----------
    results : torch.Tensor
        These contain the input and target values of the perceptron alogorithm.
        It also contains the normalised input data from which the Pytorch dataloader is created.

        It has the following keys:

        inputs : torch.Tensor
            The inputs to the perceptron algorithm, which are the outputs of the DNPU or DNPU
            architectures that you want to evaluate the accuracy against.
        norm_inputs : torch.Tensor
            Standard normal distribution of the input data. To calculate this, the
            zscore_norm function can be used in brainspy.algorithms.modules.performance.accuracy
        targets : torch.Tensor
            Binary targets against which the outuut of the perceptron algorithm is compared.

    configs : dict
        Configurations of the model to create the dataloader for the perceptron algorithm.
        The configs should contain a description of the hyperparameters that can be used
        to calculate the accuracy of the Perceptron algorithm.

        It has the following keys:

            epochs: int
                Number of loops used for training the perceptron.
            learning_rate: float
                Learning rate used to train the perceptron.
            data:
                batch size- The batch size defines the number of samples that will be propagated
                            through the network.
                pin_memory- boolean value to pin the data to memory or not.
                            If you load your samples in the Dataset on CPU and would like to push
                            it during training to the GPU,you can speed up the host to device
                            transfer by enabling pin_memory.
                            Refer to https://pytorch.org/docs/stable/data.html for batching of data
                            and memory pinning in PyTorch.
                worker_no - number of workers on the pytorch dataset which is by default set to 0.
                            Refer to https://deeplizard.com/learn/video/kWVgvsejXsE to understand
                            how the number of workers are used in deep neural network programming.

    Returns
    -------
    torch.utils.data.Dataloader
        Dataloader of the perceptron algorithm
    """
    if configs["data"]["worker_no"] > 0 or configs["data"]["pin_memory"]:
        dataset = PerceptronDataset(results["norm_inputs"],
                                    results["targets"],
                                    device=torch.device("cpu"))
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
