import torch
import unittest
import numpy as np
import matplotlib
from brainspy.algorithms.modules.performance import accuracy
from brainspy.utils.pytorch import TorchUtils


class Accuracy_Test(unittest.TestCase):
    """
    Tests to train the perceptron and calculate the accuracy
    """

    def test_get_accuracy(self):
        """
        Test for the get_accuracy method using valid input and target values
        """
        threshhold = 10000
        size = torch.randint(0, threshhold, (1, 1)).item()
        inputs = TorchUtils.format(torch.rand((size, 1)))
        targets = TorchUtils.format(torch.randint(0, 2, (size, 1)))
        results = accuracy.get_accuracy(inputs, targets)

        isinstance(results["node"], torch.nn.Linear)
        self.assertEqual(results["configs"]["epochs"], 100)
        self.assertEqual(results["configs"]["learning_rate"], 0.001)
        self.assertEqual(results["configs"]["batch_size"], 256)
        self.assertEqual(results["inputs"].shape, torch.Size((size, 1)))
        self.assertEqual(results["targets"].shape, torch.Size((size, 1)))
        self.assertTrue(results["accuracy_value"] >= 0)

    def test_get_accuracy_fail_small_dataset(self):
        """
        Test for the get_accuracy method using invalid input and target values,
        data size is too small
        """
        size = torch.randint(0, 9, (1, 1)).item()
        inputs = TorchUtils.format(torch.rand((size, 1)))
        targets = TorchUtils.format(torch.randint(0, 2, (size, 1)))
        with self.assertRaises(AssertionError):
            accuracy.get_accuracy(inputs, targets)

    def test_get_accuracy_fail_size(self):
        """
        Test for the get_accuracy method using invalid input and target values,
        inputs size != targets size
        """
        threshhold = 10000
        size = torch.randint(0, threshhold, (1, 1)).item()
        inputs = TorchUtils.format(torch.rand((size, 1)))
        targets = TorchUtils.format(torch.randint(0, 2, (size + 1, 1)))
        with self.assertRaises(RuntimeError):
            accuracy.get_accuracy(inputs, targets)

    def test_get_accuracy_fail_shape(self):
        """
        Test for the get_accuracy method using invalid input and target values,
        inputs shape != targets shape
        """
        threshhold = 10000
        size = torch.randint(0, threshhold, (1, 1)).item()
        inputs = TorchUtils.format(torch.rand((size, 2)))
        targets = TorchUtils.format(torch.randint(0, 2, (size, 1)))
        with self.assertRaises(RuntimeError):
            accuracy.get_accuracy(inputs, targets)

    def test_get_accuracy_invalid_data(self):
        """
        Invalid type for arguments raises an AssertionError
        """
        with self.assertRaises(AssertionError):
            accuracy.get_accuracy("invalid type", 100)
        with self.assertRaises(AssertionError):
            accuracy.get_accuracy([1, 2, 3, 4], torch.rand((1, 1)))
        with self.assertRaises(AssertionError):
            accuracy.get_accuracy(np.array([1, 2, 3, 4]), 100)
        with self.assertRaises(AssertionError):
            accuracy.get_accuracy(torch.rand((1, 1)), 10.10)
        with self.assertRaises(AssertionError):
            accuracy.get_accuracy(torch.rand((1, 1)), torch.rand((1, 1)), 100)

    def test_init_results(self):
        """
        Test for the init_results method to initialize the data for evaluation of accuracy
        """
        threshhold = 10000
        size = torch.randint(0, threshhold, (1, 1)).item()
        inputs = TorchUtils.format(torch.rand((size, 1)))
        targets = TorchUtils.format(torch.randint(0, 2, (size, 1)))
        configs = accuracy.get_default_node_configs()
        results, dataloader = accuracy.init_results(inputs, targets, configs)
        isinstance(dataloader, torch.utils.data.dataloader.DataLoader)
        self.assertEqual(results["inputs"].shape, torch.Size([size, 1]))
        self.assertEqual(results["targets"].shape, torch.Size([size, 1]))

    def test_init_results_invalid(self):
        """
        Invalid type for arguments raises an AssertionError
        """
        with self.assertRaises(AssertionError):
            accuracy.init_results(100, torch.rand((1, 1)), "invalid")
        with self.assertRaises(AssertionError):
            accuracy.init_results([1, 2, 3, 4], torch.rand((1, 1)), {})
        with self.assertRaises(AssertionError):
            accuracy.init_results(np.array([1, 2, 3, 4]), torch.rand((1, 1)),
                                  "invalid")
        with self.assertRaises(AssertionError):
            accuracy.init_results(torch.rand((1, 1)), torch.rand((1, 1)), 100)

    def test_zscore_norm(self):
        """
        Test for the zscore_norm method for normalization of valid input values
        """
        size = 12
        inputs = TorchUtils.format(torch.rand((size, 1)))
        val = accuracy.zscore_norm(inputs)
        self.assertEqual(val.shape, torch.Size([size, 1]))

    def test_zscore_norm_fail(self):
        """
        Test for the zscore_norm method for normalization of input values
        with standard deviation = 0
        """
        threshhold = 10000
        size = torch.randint(0, threshhold, (1, 1)).item()
        inputs = torch.ones((size, 1))
        with self.assertRaises(AssertionError):
            accuracy.zscore_norm(inputs)

    def test_zscore_invalid(self):
        """
        Invalid type for inputs raises an AssertionError
        """
        with self.assertRaises(AssertionError):
            accuracy.zscore_norm("invalid")
        with self.assertRaises(AssertionError):
            accuracy.zscore_norm([1, 2, 3, 4])
        with self.assertRaises(AssertionError):
            accuracy.zscore_norm(np.array([1, 2, 3, 4]))
        with self.assertRaises(AssertionError):
            accuracy.zscore_norm(100)

    def test_evaluate_accuracy(self):
        """
        Test to evaluate the accuracy using the perceptron algorithm
        """
        threshhold = 10000
        size = torch.randint(0, threshhold, (1, 1)).item()
        inputs = TorchUtils.format(torch.rand((size, 1)))
        targets = TorchUtils.format(torch.randint(0, 2, (size, 1)))
        node = TorchUtils.format(torch.nn.Linear(1, 1))
        accuracy_val, labels = accuracy.evaluate_accuracy(
            inputs, targets, node)
        self.assertTrue(accuracy_val > 0)

    def test_evaluate_accuracy_invalid(self):
        """
        Invalid type for inputs and targets raises an AssertionError
        """
        node = TorchUtils.format(torch.nn.Linear(1, 1))
        with self.assertRaises(AssertionError):
            accuracy.evaluate_accuracy("inputs", torch.rand((1, 1)), node)
        with self.assertRaises(AssertionError):
            accuracy.evaluate_accuracy(100, torch.rand((1, 1)), node)
        with self.assertRaises(AssertionError):
            accuracy.evaluate_accuracy(torch.rand((1, 1)), [1, 2, 3, 4], node)
        with self.assertRaises(AssertionError):
            accuracy.evaluate_accuracy(np.array([1, 2, 3, 4]),
                                       torch.rand((1, 1)), node)

    def test_evaluate_accuracy_nodetype(self):
        """
        Invalid type for node raises an AssertionError
        """
        with self.assertRaises(TypeError):
            accuracy.evaluate_accuracy(torch.rand(1, 1), torch.rand((1, 1)),
                                       "node")

    def test_train_perceptron(self):
        """
        Test to train the perceptron and check if it produces an accuracy atleast above 0%
        """
        threshhold = 10000
        size = torch.randint(0, threshhold, (1, 1)).item()
        inputs = TorchUtils.format(torch.rand((size, 1)))
        targets = TorchUtils.format(torch.randint(0, 2, (size, 1)))
        configs = accuracy.get_default_node_configs()
        results, dataloader = accuracy.init_results(inputs, targets, configs)
        node = TorchUtils.format(torch.nn.Linear(1, 1))
        optimizer = torch.optim.Adam(node.parameters(), lr=0.01)
        accuracy_val, node = accuracy.train_perceptron(130,
                                                       dataloader,
                                                       optimizer,
                                                       node=node)
        self.assertTrue(accuracy_val > 0)

    def test_train_perceptron_invalid_epoch(self):
        """
        Invalid type for epochs raises an AssertionError
        """
        threshhold = 10000
        size = torch.randint(0, threshhold, (1, 1)).item()
        inputs = TorchUtils.format(torch.rand((size, 1)))
        targets = TorchUtils.format(torch.randint(0, 2, (size, 1)))
        configs = accuracy.get_default_node_configs()
        results, dataloader = accuracy.init_results(inputs, targets, configs)
        node = TorchUtils.format(torch.nn.Linear(1, 1))
        optimizer = torch.optim.Adam(node.parameters(), lr=0.01)
        with self.assertRaises(AssertionError):
            accuracy.train_perceptron("invalid", dataloader, optimizer, node)
        with self.assertRaises(AssertionError):
            accuracy.train_perceptron([1, 2, 3, 4], dataloader, optimizer,
                                      node)
        with self.assertRaises(AssertionError):
            accuracy.train_perceptron(5.5, dataloader, optimizer, node)
        with self.assertRaises(AssertionError):
            accuracy.train_perceptron(np.array([1, 2, 3, 4]), dataloader,
                                      optimizer, node)

    def test_train_perceptron_invalid(self):
        """
        Invalid type for the arguments raises a AttributeError
        """
        with self.assertRaises(AttributeError):
            node = TorchUtils.format(torch.nn.Linear(1, 1))
            optimizer = torch.optim.Adam(node.parameters(), lr=0.01)
            accuracy.train_perceptron(100, "invalid type", optimizer, node)
        with self.assertRaises(AttributeError):
            node = TorchUtils.format(torch.nn.Linear(1, 1))
            optimizer = torch.optim.Adam(node.parameters(), lr=0.01)
            accuracy.train_perceptron(100, [1, 2, 3, 4], optimizer, node)
        with self.assertRaises(AttributeError):
            node = TorchUtils.format(torch.nn.Linear(1, 1))
            optimizer = torch.optim.Adam(node.parameters(), lr=0.01)
            accuracy.train_perceptron(100, np.array([1, 2, 3, 4]), optimizer,
                                      node)
        with self.assertRaises(AttributeError):
            threshhold = 10000
            size = torch.randint(0, threshhold, (1, 1)).item()
            inputs = TorchUtils.format(torch.rand((size, 1)))
            targets = TorchUtils.format(torch.randint(0, 2, (size, 1)))
            configs = accuracy.get_default_node_configs()
            results, dataloader = accuracy.init_results(
                inputs, targets, configs)
            node = TorchUtils.format(torch.nn.Linear(1, 1))
            accuracy.train_perceptron(100, dataloader, "invalid type", node)
        with self.assertRaises(AttributeError):
            threshhold = 10000
            size = torch.randint(0, threshhold, (1, 1)).item()
            inputs = TorchUtils.format(torch.rand((size, 1)))
            targets = TorchUtils.format(torch.randint(0, 2, (size, 1)))
            configs = accuracy.get_default_node_configs()
            results, dataloader = accuracy.init_results(
                inputs, targets, configs)
            optimizer = torch.optim.Adam(node.parameters(), lr=0.01)
            accuracy.train_perceptron(100, dataloader, optimizer,
                                      "invalid type")

    def test_default_node_configs(self):
        """
        Test to get the default node configurations for the perceptron
        """
        configs = accuracy.get_default_node_configs()
        self.assertEqual(configs["epochs"], 100)
        self.assertEqual(configs["learning_rate"], 0.001)
        self.assertEqual(configs["batch_size"], 256)

    def test_plot_perceptron(self):
        """
        Test to plot the perceppton which returns a figure which is an instance of the
        matplotlib library
        """
        threshhold = 10000
        size = torch.randint(0, threshhold, (1, 1)).item()
        inputs = TorchUtils.format(torch.rand((size, 1)))
        targets = TorchUtils.format(torch.randint(0, 2, (size, 1)))
        results = accuracy.get_accuracy(inputs, targets)
        fig = accuracy.plot_perceptron(results)
        self.assertTrue(fig, matplotlib.pyplot.figure)

    def test_plot_perceptron_invalid_type(self):
        """
        Invalid type for results raises an AssertionErrror
        """
        with self.assertRaises(AssertionError):
            accuracy.plot_perceptron("invalid type")
        with self.assertRaises(AssertionError):
            accuracy.plot_perceptron(100)
        with self.assertRaises(AssertionError):
            accuracy.plot_perceptron(np.array([1, 2, 3, 4]))
        with self.assertRaises(AssertionError):
            accuracy.plot_perceptron([1, 2, 3, 4])


if __name__ == "__main__":
    unittest.main()
