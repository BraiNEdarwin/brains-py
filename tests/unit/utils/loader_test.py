"""
Placeholder module docstring.
"""

import unittest
from collections import OrderedDict
import torch
from brainspy.processors.simulation.processor import load_file


class LoaderTest(unittest.TestCase):
    """
    Class for testing 'loader.py'.
    """

    def test_all(self):
        """
        Test if loader makes the necessary adjustments.
        """
        path = "model.pt"
        altered_path = "altered_model.pt"
        model = torch.load(path)  # Manually loaded model.

        # Make an altered model that does not have the required settings and save it.
        if "amplification" in model["info"]["data_info"]["processor"]:
            del model["info"]["data_info"]["processor"]["amplification"]
        if "D_in" in model["info"]["smg_configs"]["processor"]["torch_model_dict"]:
            del model["info"]["smg_configs"]["processor"]["torch_model_dict"]["D_in"]
        if "D_out" in model["info"]["smg_configs"]["processor"]["torch_model_dict"]:
            del model["info"]["smg_configs"]["processor"]["torch_model_dict"]["D_out"]
        if (
            "hidden_sizes"
            in model["info"]["smg_configs"]["processor"]["torch_model_dict"]
        ):
            del model["info"]["smg_configs"]["processor"]["torch_model_dict"][
                "hidden_sizes"
            ]
        torch.save(model, altered_path)

        # Load the altered model with "loader".
        info, state_dict = load_file(altered_path)

        # Check if state_dict is right type.
        self.assertIsInstance(state_dict, OrderedDict)

        # Check if amplification is set to 1.
        self.assertFalse(
            "amplification" in model["info"]["data_info"]["processor"].keys()
        )
        self.assertTrue("amplification" in info["data_info"]["processor"].keys())
        self.assertEqual(info["data_info"]["processor"]["amplification"], 1)

        # Check if consistency check sets "D_in", "D_out" and "hidden_sizes".
        self.assertFalse(
            "D_in"
            in model["info"]["smg_configs"]["processor"]["torch_model_dict"].keys()
        )
        self.assertTrue("D_out" in info["smg_configs"]["processor"]["torch_model_dict"])
        self.assertFalse(
            "D_out"
            in model["info"]["smg_configs"]["processor"]["torch_model_dict"].keys()
        )
        self.assertTrue(
            "hidden_sizes" in info["smg_configs"]["processor"]["torch_model_dict"]
        )
        self.assertFalse(
            "D_in"
            in model["info"]["smg_configs"]["processor"]["torch_model_dict"].keys()
        )
        self.assertTrue(
            "hidden_sizes" in info["smg_configs"]["processor"]["torch_model_dict"]
        )


if __name__ == "__main__":
    unittest.main()
