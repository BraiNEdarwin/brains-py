import os
import torch
import unittest
import brainspy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from brainspy.algorithms.modules.performance.accuracy import *
from brainspy.utils.pytorch import TorchUtils


class Accuracy_Test(unittest.TestCase):
    """
    Tests to train the perceptron and calculate the accuracy
    """
    def __init__(self, test_name):
        super(Accuracy_Test, self).__init__()

    def test_get_accuracy(self):
        """
        Test for the get_accuracy method using valid input and target values
        """
        size = torch.randint(11, 200, (1, 1)).item()
        inputs = TorchUtils.format(torch.rand((size, 1)))
        targets = TorchUtils.format(torch.randint(0, 2, (size, 1)))
        # inputs = torch.tensor([
        #     [-0.9650],
        #     [-0.9650],
        #     [1.1565],
        #     [1.1565],
        #     [-0.346],
        #     [0.7145],
        #     [0.2726],
        #     [0.2726],
        #     [-1.6721],
        #     [0.8913],
        #     [-1.2478],
        #     [0.73225],
        # ])
        # targets = torch.tensor([
        #     [0.0],
        #     [0.0],
        #     [0.0],
        #     [1.0],
        #     [0.0],
        #     [0.0],
        #     [0.0],
        #     [0.0],
        #     [0.0],
        #     [0.0],
        #     [1.0],
        #     [1.0],
        # ])
        # inputs = TorchUtils.format(inputs, device=TorchUtils.get_device())
        # targets = TorchUtils.format(targets, device=TorchUtils.get_device())
        results = get_accuracy(inputs, targets)

        isinstance(results["node"], torch.nn.Linear)
        self.assertEqual(results["configs"]["epochs"], 100)
        self.assertEqual(results["configs"]["learning_rate"], 0.001)
        self.assertEqual(results["configs"]["batch_size"], 256)
        self.assertEqual(results["inputs"].shape, torch.Size((size, 1)))
        self.assertEqual(results["targets"].shape, torch.Size((size, 1)))
        #self.assertEqual(results["inputs"].shape, torch.Size([12, 1]))
        #self.assertEqual(results["targets"].shape, torch.Size([12, 1]))
        #self.assertTrue(results["accuracy_value"] >= 0)

    def test_get_accuracy_fail(self):
        """
        Test for the get_accuracy method using invalid input and target values, data size is too small
        """
        inputs = torch.tensor([
            [-0.9650],
            [-0.9650],
            [1.1565],
            [1.1565],
            [-0.346],
            [0.7145],
            [0.2726],
            [0.2726],
        ])
        targets = torch.tensor([[0.0], [0.0], [0.0], [1.0], [0.0], [0.0],
                                [0.0], [0.0]])
        inputs = TorchUtils.format(inputs, device=TorchUtils.get_device())
        targets = TorchUtils.format(targets, device=TorchUtils.get_device())

        raised = False
        try:
            get_accuracy(inputs, targets)
        except AssertionError:
            raised = True
        self.assertTrue(raised,
                        "Not enough data, at least 10 points are required")

    def test_get_accuracy_fail_2(self):
        """
        Test for the get_accuracy method using invalid input and target values, inputs size != targets size
        """
        inputs = torch.tensor([
            [-0.9650],
            [-0.9650],
            [1.1565],
            [1.1565],
            [-0.346],
            [0.7145],
            [0.2726],
            [0.2726],
            [-1.6721],
            [0.8913],
            [-1.2478],
            [0.73225],
        ])
        targets = torch.tensor([[0.0], [0.0], [0.0], [1.0], [0.0], [0.0],
                                [0.0], [0.0]])
        inputs = TorchUtils.format(inputs, device=TorchUtils.get_device())
        targets = TorchUtils.format(targets, device=TorchUtils.get_device())

        raised = False
        try:
            get_accuracy(inputs, targets)
        except IndexError:
            raised = True
        self.assertTrue(raised, "Unequal number of inputs and targets")

    def test_get_accuracy_fail_3(self):
        """
        Test for the get_accuracy method using invalid input and target values, inputs shape != targets shape
        """
        inputs = torch.tensor([
            [-1.0234, -1.0234],
            [-1.0234, 1.1124],
            [1.1124, -1.0234],
            [1.1124, 1.1124],
            [-0.4005, 0.2225],
            [0.6674, 0.2225],
            [0.2225, -0.4005],
            [0.2225, 0.6674],
            [-1.7353, 0.8454],
            [0.8454, -1.7353],
            [0.5, 0.5],
        ])
        targets = torch.tensor([
            [0.0],
            [0.0],
            [0.0],
            [1.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.1],
        ])
        inputs = TorchUtils.format(inputs, device=TorchUtils.get_device())
        targets = TorchUtils.format(targets, device=TorchUtils.get_device())
        raised = False
        try:
            get_accuracy(inputs, targets)
        except RuntimeError:
            raised = True
        self.assertTrue(raised, "Shapes of inputs and targets do not match")

    def test_init_results(self):
        inputs = torch.tensor([
            [-0.9650],
            [-0.9650],
            [1.1565],
            [1.1565],
            [-0.346],
            [0.7145],
            [0.2726],
            [0.2726],
            [-1.6721],
            [0.8913],
            [-1.2478],
            [0.73225],
        ])
        targets = torch.tensor([
            [0.0],
            [0.0],
            [0.0],
            [1.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [1.0],
            [1.0],
        ])
        inputs = TorchUtils.format(inputs, device=TorchUtils.get_device())
        targets = TorchUtils.format(targets, device=TorchUtils.get_device())
        configs = get_default_node_configs()
        results, dataloader = init_results(inputs, targets, configs)
        isinstance(dataloader, torch.utils.data.dataloader.DataLoader)
        self.assertEqual(results["inputs"].shape, torch.Size([12, 1]))
        self.assertEqual(results["targets"].shape, torch.Size([12, 1]))

    def test_zscore_norm(self):
        """
        Test for the zscore_norm method for normalization of valid input values
        """
        inputs = torch.tensor([
            [-0.9650],
            [-0.9650],
            [1.1565],
            [1.1565],
            [-0.346],
            [0.7145],
            [0.2726],
            [0.2726],
            [-1.6721],
            [0.8913],
            [-1.2478],
            [0.73225],
        ])
        val = zscore_norm(inputs)
        self.assertEqual(val.shape, torch.Size([12, 1]))

    def test_zscore_norm_fail(self):
        """
        Test for the zscore_norm method for normalization of input values with standard deviation = 0
        """
        inputs = torch.tensor([
            [1.0],
            [1.0],
            [1.0],
            [1.0],
            [1.0],
            [1.0],
            [1.0],
            [1.0],
            [1.0],
            [1.0],
            [1.0],
            [1.0],
        ])
        raised = False
        try:
            val = zscore_norm(inputs)
        except AssertionError:
            raised = True
        self.assertTrue(raised, "Standard deviation of input data is 0")

    def test_evaluate_accuracy(self):
        """
        Test to evaluate the accuracy using the perceptron algorithm
        """
        inputs = torch.tensor([
            [-0.9650],
            [-0.9650],
            [1.1565],
            [1.1565],
            [-0.346],
            [0.7145],
            [0.2726],
            [0.2726],
            [-1.6721],
            [0.8913],
            [-1.2478],
            [0.73225],
        ])
        targets = torch.tensor([
            [0.0],
            [0.0],
            [0.0],
            [1.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [1.0],
            [1.0],
        ])
        inputs = TorchUtils.format(inputs, device=TorchUtils.get_device())
        targets = TorchUtils.format(targets, device=TorchUtils.get_device())
        node = TorchUtils.format(torch.nn.Linear(1, 1))
        accuracy, labels = evaluate_accuracy(inputs, targets, node)
        self.assertTrue(accuracy > 0)

    def test_train_perceptron(self):
        """
        Test to train the perceptron and check if it produces an accuracy atleast above 0%
        """
        inputs = torch.tensor([
            [-0.9650],
            [-0.9650],
            [1.1565],
            [1.1565],
            [-0.346],
            [0.7145],
            [0.2726],
            [0.2726],
            [-1.6721],
            [0.8913],
            [-1.2478],
            [0.73225],
        ])
        targets = torch.tensor([
            [0.0],
            [0.0],
            [0.0],
            [1.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [1.0],
            [1.0],
        ])
        configs = get_default_node_configs()
        results, dataloader = init_results(inputs, targets, configs)
        node = TorchUtils.format(torch.nn.Linear(1, 1))
        optimizer = torch.optim.Adam(node.parameters(), lr=0.01)
        accuracy, node = train_perceptron(130,
                                          dataloader,
                                          optimizer,
                                          node=node)
        self.assertTrue(accuracy > 0)

    def test_default_node_configs(self):
        """
        Test to get the default node configurations for the perceptron
        """
        configs = get_default_node_configs()
        self.assertEqual(configs["epochs"], 100)
        self.assertEqual(configs["learning_rate"], 0.001)
        self.assertEqual(configs["batch_size"], 256)

    def test_plot_perceptron(self):
        """
        Test to plot the perceppton which returns a figure which is an instance of the matplotlib library
        """
        inputs = torch.tensor([
            [-0.9650],
            [-0.9650],
            [1.1565],
            [1.1565],
            [-0.346],
            [0.7145],
            [0.2726],
            [0.2726],
            [-1.6721],
            [0.8913],
            [-1.2478],
            [0.73225],
        ])
        targets = torch.tensor([
            [0.0],
            [0.0],
            [0.0],
            [1.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [1.0],
            [1.0],
        ])
        inputs = TorchUtils.format(inputs, device=TorchUtils.get_device())
        targets = TorchUtils.format(targets, device=TorchUtils.get_device())
        results = get_accuracy(inputs, targets)
        fig = plot_perceptron(results)
        self.assertTrue(fig, matplotlib.pyplot.figure)

    def runTest(self):
        self.test_get_accuracy()
        self.test_get_accuracy_fail()
        self.test_get_accuracy_fail_2()
        self.test_get_accuracy_fail_3()
        self.test_init_results()
        self.test_zscore_norm()
        self.test_zscore_norm_fail()
        self.test_evaluate_accuracy()
        self.test_train_perceptron()
        self.test_default_node_configs()
        self.test_plot_perceptron()


if __name__ == "__main__":
    unittest.main()
