"""
Module for testing 'processor.py'.
"""

import unittest
import warnings

import torch
import numpy as np

from brainspy.utils.pytorch import TorchUtils
from brainspy.processors.simulation.processor import SurrogateModel
from brainspy.processors.simulation.noise.noise import GaussianNoise


class ModelTest(unittest.TestCase):
    """
    Class for testing 'processor.py'.
    """
    def setUp(self):
        """
        Make a SurrogateModel and info dictionary to work with.
        """
        model_structure = {
            "D_in": 7,
            "D_out": 1,
            "activation": "relu",
            "hidden_sizes": [20, 20, 20]
        }
        self.model = TorchUtils.format(SurrogateModel(model_structure))
        self.info_dict = {
            'activation_electrodes': {
                'electrode_no': 7,
                'voltage_ranges': [[1.0, 2.0]] * 7
            },
            'output_electrodes': {
                'electrode_no': 1,
                'amplification': [28.5],
                'clipping_value': [-114.0, 114.0]
            }
        }
        self.model.set_effects_from_dict(self.info_dict, dict())

    def test_get_voltage_ranges(self):
        """
        Test if setting the voltage ranges works, both for default setting and
        with a value.
        """
        # should set to configs
        configs = {"voltage_ranges": [[3.0, 4.0]] * 7}
        self.model.set_effects_from_dict(self.info_dict, configs)
        self.assertTrue(
            torch.equal(TorchUtils.format(self.model.get_voltage_ranges()),
                        TorchUtils.format([[3.0, 4.0]] * 7)))

        # should set to default from info dict
        self.model.set_effects_from_dict(self.info_dict, {})
        self.assertTrue(
            torch.equal(TorchUtils.format(self.model.get_voltage_ranges()),
                        TorchUtils.format([[1.0, 2.0]] * 7)))

    def test_forward(self):
        """
        Test if a forward pass through the processor returns a tensor of the
        right shape.
        """
        for i in range(100):
            x = TorchUtils.format(torch.rand(7))
            x = self.model.forward(x)
            self.assertEqual(list(x.shape), [1])

    def test_forward_numpy(self):
        """
        Test if a forward pass through the processor returns a tensor of the
        right shape (the numpy version).
        """
        for i in range(100):
            x = np.array(np.random.random(7))
            x = self.model.forward_numpy(x)
            self.assertEqual(list(x.shape), [1])

    def test_close(self):
        """
        Test if closing the processor raises a warning.
        """
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            self.model.close()
            self.assertEqual(len(caught_warnings), 1)

    def test_is_hardware(self):
        """
        Test if processor knows it is not hardware.
        """
        self.assertFalse(self.model.is_hardware())

    def test_set_effects_from_dict(self):
        """
        Test setting effects from a dictionary.
        """
        configs = {
            "amplification": None,
            "output_clipping": [4.0, 3.0],
            "voltage_ranges": "default",
            "noise": None,
            "test": 0
        }
        self.model.set_effects_from_dict(self.info_dict, configs)
        self.assertIsNone(self.model.amplification)
        self.assertTrue(
            torch.equal(TorchUtils.format(self.model.output_clipping),
                        TorchUtils.format([4.0, 3.0])))
        self.assertTrue(
            torch.equal(
                TorchUtils.format(self.model.get_voltage_ranges()),
                TorchUtils.format(self.info_dict["activation_electrodes"]
                                  ["voltage_ranges"])))

    def test_get_key(self):
        """
        Test if get key method returns the key if it exists, and "default" or
        None if it does not exist.
        """
        key1 = "key1"
        key2 = "key2"
        key_noise = "noise"
        d = {"key1": 1}
        self.assertEquals(self.model.get_key(d, key1), 1)
        self.assertEquals(self.model.get_key(d, key2), "default")
        self.assertEquals(self.model.get_key(d, key_noise), "default")

    def test_set_effects(self):
        """
        Test setting effects.
        """
        self.model.set_effects(self.info_dict,
                               amplification=[3.0],
                               voltage_ranges="default",
                               output_clipping=np.array([4.0, 3.0]))
        print(self.model.output_clipping)
        self.assertEquals(TorchUtils.format(self.model.amplification),
                          TorchUtils.format([3.0]))
        self.assertTrue(
            torch.equal(TorchUtils.format(self.model.output_clipping),
                        TorchUtils.format([4.0, 3.0]).double()))
        self.assertTrue(
            torch.equal(
                TorchUtils.format(self.model.get_voltage_ranges()),
                TorchUtils.format(self.info_dict["activation_electrodes"]
                                  ["voltage_ranges"])))

    def test_set_voltage_ranges(self):
        """
        Test setting voltage ranges to default and new value.
        """
        # set to value
        value = [[3.0, 4.0]] * 7
        self.model.set_voltage_ranges(self.info_dict, value)
        self.assertTrue(
            torch.equal(TorchUtils.format(self.model.get_voltage_ranges()),
                        TorchUtils.format(value)))

        # set to default
        self.model.set_voltage_ranges(self.info_dict, "default")
        self.assertTrue(
            torch.equal(
                TorchUtils.format(self.model.get_voltage_ranges()),
                TorchUtils.format(self.info_dict["activation_electrodes"]
                                  ["voltage_ranges"])))

    def test_set_amplification(self):
        """
        Test setting amplification to default, None or a new value.
        """
        # set to value
        value = [3.0]
        self.model.set_amplification(self.info_dict, value)
        self.assertTrue(
            torch.equal(TorchUtils.format(self.model.amplification),
                        TorchUtils.format(value)))

        # set to default
        self.model.set_amplification(self.info_dict, "default")
        self.assertTrue(
            torch.equal(
                TorchUtils.format(self.model.amplification),
                TorchUtils.format(
                    self.info_dict["output_electrodes"]["amplification"])))

        # set to None
        self.model.set_amplification(self.info_dict, None)
        self.assertIsNone(self.model.amplification)

    def test_set_output_clipping(self):
        """
        Test setting output clipping to default, None or a new value.
        """
        # set to value
        value = [4.0, 3.0]
        self.model.set_output_clipping(self.info_dict, value)
        self.assertTrue(
            torch.equal(TorchUtils.format(self.model.output_clipping),
                        TorchUtils.format(value)))

        # set to default
        self.model.set_output_clipping(self.info_dict, "default")
        self.assertTrue(
            torch.equal(
                TorchUtils.format(self.model.output_clipping),
                TorchUtils.format(
                    self.info_dict["output_electrodes"]["clipping_value"])))

        # set to None
        self.model.set_output_clipping(self.info_dict, None)
        self.assertIsNone(self.model.output_clipping)

    def test_set_noise(self):
        """
        Test setting the noise to gaussian or None.
        """
        # set to none
        self.model.set_effects(self.info_dict)
        self.assertIsNone(self.model.noise)

        # set to Gaussian
        noise_dict = {"type": "gaussian", "variance": 1.0}
        self.model.set_effects(self.info_dict, noise_configs=noise_dict)
        self.assertIsInstance(self.model.noise, GaussianNoise)


if __name__ == "__main__":
    unittest.main()
