"""
Module for testing 'model.py'.
"""

import unittest
import warnings

import torch
from torch import nn

from brainspy.processors.simulation.model import NeuralNetworkModel


class ModelTest(unittest.TestCase):
    """
    Class for testing 'model.py'.
    """
    def setUp(self):
        """
        Create a model to work with. The info consistency check will
        automatically create the dictionary.
        """
        self.model = NeuralNetworkModel({})

    def test_verbose(self):
        """
        Test that model works with verbose option on.
        """
        NeuralNetworkModel({}, verbose=True)

    def test_build_model_structure(self):
        """
        Test build model structure.
        """
        self.model.build_model_structure({})
        raw = self.model.raw_model
        # input layer, 6 activations, 5 hidden layers, output layer
        self.assertEqual(len(raw), 13)

    # def test_info_dict(self):
    #     """
    #     Test setting and getting info dict.
    #     """
    #     info_dict = {"test1": 1, "test2": 2}
    #     self.model.set_info_dict(info_dict)
    #     self.assertEqual(info_dict, self.model.get_info_dict())

    def test_get_activation(self):
        """
        Test get activation.
        """
        self.assertIsInstance(self.model._get_activation("relu"), nn.ReLU)
        self.assertIsInstance(self.model._get_activation("elu"), nn.ELU)
        self.assertIsInstance(self.model._get_activation("tanh"), nn.Tanh)
        self.assertIsInstance(self.model._get_activation("hard-tanh"),
                              nn.Hardtanh)
        self.assertIsInstance(self.model._get_activation("sigmoid"),
                              nn.Sigmoid)
        with warnings.catch_warnings(record=True) as caught_warnings:
            # If string is not recognized the method should return relu and
            # raise a warning.
            warnings.simplefilter("always")
            self.assertIsInstance(self.model._get_activation("test"), nn.ReLU)
            self.assertEqual(len(caught_warnings), 1)

    def test_forward(self):
        """
        Test the forward pass, check whether result has right shape.
        """
        x = torch.rand((7))
        result = self.model.forward(x)
        test = torch.rand((1))
        self.assertEqual(result.shape, test.shape)

    def test_consistency_check(self):
        """
        Test if info_consistency_check makes the necessary adjustments.
        """

        # Check if consistency check sets "D_in", "D_out" and "hidden_sizes".
        # Make sure warnings are thrown if a key is missing.
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            d = {}
            self.model.structure_consistency_check(d)
            self.assertTrue("D_out" in d)
            self.assertTrue("hidden_sizes" in d)
            self.assertTrue("activation" in d)
            self.assertEqual(len(caught_warnings), 4)


if __name__ == "__main__":
    unittest.main()
