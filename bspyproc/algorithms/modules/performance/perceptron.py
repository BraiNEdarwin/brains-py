import torch
from torch.utils.data import Dataset


class Perceptron(torch.nn.Module):
    def __init__(self, activation=torch.nn.Sigmoid()):
        super(Perceptron, self).__init__()
        self.linear = torch.nn.Linear(1, 1)
        self.activation = activation

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        return x


class PerceptronDataset(Dataset):

    # TODO: use data object to get the accuracy (see corr_coeff above)
    def __init__(self, inputs, targets):
        # Normalise inputs
        assert len(inputs) > 10, "Not enough data, at least 10 points are required."
        self.inputs = inputs
        self.targets = targets

    def __getitem__(self, index):
        inputs = self.inputs[index, :]
        targets = self.targets[index, :]

        return (inputs, targets)
