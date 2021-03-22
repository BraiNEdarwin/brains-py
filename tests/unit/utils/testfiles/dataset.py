import numpy as np
from torch.utils.data import Dataset

# Sample dataset class used to test the manager.py util
X = [-0.5, -0.5, 1.0, 1.0, -0.0625, 0.6875, 0.375, 0.375, -1.0, 0.8125]
Y = [-0.5, 1.0, -0.5, 1.0, 0.375, 0.375, -0.0625, 0.6875, 0.8125, -1.0]


class BooleanGateDataset(Dataset):
    def __init__(self, target, transforms=None, verbose=True):
        self.transforms = transforms
        self.inputs = self.generate_inputs(len(target))
        self.targets = target.T[:, np.newaxis]

    def __getitem__(self, index):
        inputs = self.inputs[index, :]
        targets = self.targets[index, :]
        sample = (inputs, targets)
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample

    def __len__(self):
        return len(self.targets)

    def generate_inputs(self, vc_dimension):
        assert len(X) == len(
            Y
        ), f"Number of data in both dimensions must be equal ({len(X)},{len(Y)})"
        assert vc_dimension <= len(
            X
        ), "VC Dimension exceeds the current number of points"
        return np.array([X[:vc_dimension], Y[:vc_dimension]]).T
