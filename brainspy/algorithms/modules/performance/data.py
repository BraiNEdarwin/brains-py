import torch
from torch.utils.data import Dataset, random_split

DTYPE = torch.float32


def get_data(results, configs, shuffle=True):
    # Prepare perceptron data
    if configs["data"]["worker_no"] > 0 or configs["data"]["pin_memory"]:
        dataset = PerceptronDataset(
            results["norm_inputs"], results["targets"], device=torch.device("cpu")
        )
    else:
        dataset = PerceptronDataset(results["norm_inputs"], results["targets"])

    if configs["data"]["split"][1] != 0:
        lengths = [
            len(dataset) * configs["split"][0],
            len(dataset) * configs["split"][1],
        ]

        subsets = random_split(dataset, lengths)
        dataloaders = [
            torch.utils.data.DataLoader(
                subsets[i],
                batch_size=configs["data"]["batch_size"],
                shuffle=shuffle,
                num_workers=configs["data"]["worker_no"],
                pin_memory=configs["data"]["pin_memory"],
            )
            for i in range(len(subsets))
        ]
    else:
        dataloaders = [
            torch.utils.data.DataLoader(
                dataset,
                batch_size=configs["data"]["batch_size"],
                shuffle=shuffle,
                num_workers=configs["data"]["worker_no"],
                pin_memory=configs["data"]["pin_memory"],
            )
        ]

    return dataloaders


class PerceptronDataset(Dataset):

    # TODO: use data object to get the accuracy (see corr_coeff above)
    def __init__(self, inputs, targets, device=None):

        # Normalise inputs
        assert len(inputs) > 10, "Not enough data, at least 10 points are required."
        assert not torch.isnan(inputs).any(), "NaN values detected."
        if device is None:
            self.inputs = inputs.to(dtype=DTYPE)
            self.targets = targets.to(dtype=DTYPE)
        else:
            self.inputs = inputs.to(device=device, dtype=DTYPE)
            self.targets = targets.to(device=device, dtype=DTYPE)

    def __getitem__(self, index):
        inputs = self.inputs[index, :]
        targets = self.targets[index, :]

        return (inputs, targets)

    def __len__(self):
        return len(self.inputs)
