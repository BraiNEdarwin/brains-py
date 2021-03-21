import numpy as np
from torch.utils.data import Dataset

# Set of points in a scale from -1 to 1
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


def generate_targets(vc_dimension, verbose=True):
    # length of list, i.e. number of binary targets
    binary_target_no = 2 ** vc_dimension
    assignments = []
    list_buf = []

    # construct assignments per element i
    if verbose:
        print("===" * vc_dimension)
        print("ALL BINARY LABELS:")
    level = int((binary_target_no / 2))
    while level >= 1:
        list_buf = []
        buf0 = [0] * level
        buf1 = [1] * level
        while len(list_buf) < binary_target_no:
            list_buf += buf0 + buf1
        assignments.append(list_buf)
        level = int(level / 2)

    binary_targets = np.array(assignments).T
    if verbose:
        print(binary_targets)
        print("===" * vc_dimension)
    return binary_targets[1:-1]  # Remove [0,0,0,0] and [1,1,1,1] gates
