import unittest
import warnings

import torch

from brainspy.processors.simulation.model import NeuralNetworkModel


class ModelTest(unittest.TestCase):
    """
    Class for testing 'model.py'.
    """
    def test_consistency_check(self):
        """
        Test if info_consistency_check makes the necessary adjustments.
        """
        path = "brains-py/model.pt"

        # Load a model.
        model = torch.load(
            path)["info"]["smg_configs"]["processor"]["torch_model_dict"]

        nn = NeuralNetworkModel(model)

        # Delete needed keys.
        if "D_in" in model:
            del model["D_in"]
        if "D_out" in model:
            del model["D_out"]
        if "hidden_sizes" in model:
            del model["hidden_sizes"]
        if "activation" in model:
            del model["activation"]

        # Check if consistency check sets "D_in", "D_out" and "hidden_sizes".
        # Make sure warnings are thrown if a key is missing.
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            nn.info_consistency_check(model)
            self.assertTrue("D_out" in model)
            self.assertTrue("hidden_sizes" in model)
            self.assertTrue("activation" in model)
            self.assertTrue(model["activation"] in ("relu", "elu"))
            self.assertEqual(len(caught_warnings), 4)


if __name__ == "__main__":
    unittest.main()
