import os
import torch
import unittest
import numpy as np
import brainspy
from brainspy.algorithms.modules.performance.accuracy import zscore_norm
from brainspy.algorithms.modules.performance.data import get_data


class Data_Test(unittest.TestCase):

    """
    Tests for the Perceptron dataloader - data.py.

    """

    def __init__(self, test_name):
        super(Data_Test, self).__init__()

    def test_get_data(self):
        """
        Test to get data from the Perceptron dataloader
        """
        inputs = torch.rand(100)
        targets = torch.rand(100)
        results = {}
        results["inputs"] = inputs
        results["targets"] = targets
        results["norm_inputs"] = zscore_norm(inputs)

        configs = {}
        configs["data"] = {}
        configs["data"]["pin_memory"] = True
        configs["data"]["batch_size"] = 512
        configs["data"]["worker_no"] = 0
        configs["epochs"] = 130
        configs["learning_rate"] = 0.01

        dataloader = get_data(results, configs)

        self.assertEqual(dataloader.pin_memory, True)
        self.assertEqual(dataloader.batch_size, 512)
        self.assertEqual(dataloader.num_workers, 0)
        self.assertEqual(dataloader.drop_last, False)

    def test_get_data1(self):
        """
        Test for assertion error if input size is too small
        """
        inputs = torch.rand(5)
        targets = torch.rand(5)
        results = {}
        results["inputs"] = inputs
        results["targets"] = targets
        results["norm_inputs"] = zscore_norm(inputs)

        configs = {}
        configs["data"] = {}
        configs["data"]["pin_memory"] = True
        configs["data"]["batch_size"] = 512
        configs["data"]["worker_no"] = 0
        configs["epochs"] = 130
        configs["learning_rate"] = 0.01

        try:
            get_data(results, configs)
        except:
            AssertionError

    def test_get_data2(self):
        """
        Test for assertion error if NaN values detected in the input data
        """
        inputs = torch.tensor(
            [
                1,
                float("nan"),
                2,
                1,
                float("nan"),
                2,
                1,
                float("nan"),
                2,
                1,
                float("nan"),
                2,
                1,
                float("nan"),
                2,
                1,
                float("nan"),
                2,
                1,
                float("nan"),
                2,
                1,
                float("nan"),
                2,
                1,
                float("nan"),
                2,
            ]
        )
        targets = torch.rand(5)
        results = {}
        results["inputs"] = inputs
        results["targets"] = targets
        results["norm_inputs"] = zscore_norm(inputs)

        configs = {}
        configs["data"] = {}
        configs["data"]["pin_memory"] = True
        configs["data"]["batch_size"] = 512
        configs["data"]["worker_no"] = 0
        configs["epochs"] = 130
        configs["learning_rate"] = 0.01

        try:
            get_data(results, configs)
        except:
            AssertionError

    def runTest(self):
        self.test_get_data()
        self.test_get_data1()
        self.test_get_data2()


if __name__ == "__main__":
    unittest.main()
