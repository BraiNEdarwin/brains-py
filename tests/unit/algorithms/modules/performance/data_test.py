import torch
import unittest
import random
import numpy as np
from brainspy.utils.pytorch import TorchUtils
from brainspy.algorithms.modules.performance.accuracy import zscore_norm
from brainspy.algorithms.modules.performance.data import get_data, PerceptronDataset


class Data_Test(unittest.TestCase):
    """
    Tests for the Perceptron dataloader - data.py.

    """
    def test_get_data(self):
        """
        Test to get data from the Perceptron dataloader with some random input and target values
        """
        results = {}
        results["inputs"] = TorchUtils.format(torch.rand(1000))
        results["targets"] = TorchUtils.format(torch.rand(1000))
        results["norm_inputs"] = zscore_norm(results["inputs"])
        try:
            dataloader = get_data(results, batch_size=512)
            self.assertEqual(dataloader.batch_size, 512)
            self.assertEqual(dataloader.drop_last, False)
        except (Exception):
            self.fail("Could not get data from the Perceptron Dataloader")

    def test_get_data_small(self):
        """
        AssertionError should be raised if the dataset is too small
        """
        results = {}
        results["inputs"] = TorchUtils.format(torch.rand(5))
        results["targets"] = TorchUtils.format(torch.rand(5))
        results["norm_inputs"] = zscore_norm(results["inputs"])

        with self.assertRaises(AssertionError):
            get_data(results, batch_size=512)

    def test_get_data_nan(self):
        """
        AssertionError is raised if the input dataset contains nan values
        """
        results = {}
        results["inputs"] = torch.tensor([
            1,
            float("nan"),
            2,
            1,
        ])
        results["targets"] = TorchUtils.format(torch.rand(5))
        results["norm_inputs"] = zscore_norm(results["inputs"])

        with self.assertRaises(AssertionError):
            get_data(results, batch_size=512)

    def test_get_data_invalid_dtype(self):
        """
        AssertionError should be raised if the input is of incorrect type
        """
        with self.assertRaises(AssertionError):
            get_data("Inavlid type", "Invalid type")
        with self.assertRaises(AssertionError):
            get_data({"key": "Inavlid type"}, "Invalid type")
        with self.assertRaises(AssertionError):
            get_data(1, {"key": "Invalid type"})
        with self.assertRaises(AssertionError):
            get_data(5.6, [1, 2, 3, 4])
        with self.assertRaises(AssertionError):
            get_data("Inavlid type", 100)
        with self.assertRaises(AssertionError):
            get_data(np.array([1, 2, 3, 4]), "Invalid type")

    def test_get_data_invalid_key_type(self):
        """
        AssertionError should be raised if the individual
        keys are of the wrong type
        """
        results = {}
        results["inputs"] = "invalid type"
        results["targets"] = TorchUtils.format(torch.rand(500))
        results["norm_inputs"] = zscore_norm(TorchUtils.format(
            torch.rand(500)))

        with self.assertRaises(AssertionError):
            get_data(results, batch_size=512)

        results = {}
        results["inputs"] = TorchUtils.format(torch.rand(500))
        results["targets"] = [1, 2, 3, 4]
        results["norm_inputs"] = zscore_norm(results["inputs"])

        with self.assertRaises(AssertionError):
            get_data(results, batch_size=512)

    def test_get_data_key_missing(self):
        """
        A KeyError is raised if a key is missing in the input dict
        Here, missing key - "norm_inputs" in the results dict
        """
        results = {}
        results["inputs"] = TorchUtils.format(torch.rand(500))

        with self.assertRaises(KeyError):
            get_data(results, batch_size=512)

    def test_get_data_batch_size(self):
        """
        Test to get_data from the Perceptron Dataloader with a batch size
        of a random value
        """
        results = {}
        results["inputs"] = TorchUtils.format(torch.rand(1000))
        results["targets"] = TorchUtils.format(torch.rand(1000))
        results["norm_inputs"] = zscore_norm(results["inputs"])
        try:
            get_data(results, batch_size=random.randint(0, 1000))
        except (Exception):
            self.fail(
                "Could not get data from the Perceptron Dataloader with this batch size"
            )

    def test_get_data_batch_size_negative(self):
        """
        AssertionError is raised if a negative value for batch_size is provided
        """
        results = {}
        results["inputs"] = TorchUtils.format(torch.rand(1000))
        results["targets"] = TorchUtils.format(torch.rand(1000))
        results["norm_inputs"] = zscore_norm(results["inputs"])
        with self.assertRaises(AssertionError):
            get_data(results, batch_size=random.randint(-1000, -1))

    def test_PerceptronDatasetclass(self):
        """
        Test for the PerceptronDataset Class
        """
        results = {}
        results["inputs"] = TorchUtils.format(torch.rand(1000))
        results["targets"] = TorchUtils.format(torch.rand(1000))
        results["norm_inputs"] = zscore_norm(results["inputs"])
        try:
            dataset = PerceptronDataset(results["norm_inputs"],
                                        results["targets"])
            self.assertEqual(len(results["norm_inputs"]), len(dataset))
        except (Exception):
            self.fail("Could not initialize PerceptronDataset class")


if __name__ == "__main__":
    unittest.main()
