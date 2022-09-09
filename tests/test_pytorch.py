"""
Module for testing the management of torch variables.
"""
import unittest
import numpy as np
import torch
from brainspy.utils.pytorch import TorchUtils


class PyTorchTest(unittest.TestCase):
    """
    Tests for TorchUtils in the pytorch.py class.
    """
    def setUp(self):
        self.device = TorchUtils.get_device()

    def test_set_force_cpu_true(self):
        """
        Test for the set_force_cpu() method to set it to True
        """
        TorchUtils.set_force_cpu(True)
        self.assertEqual(TorchUtils.force_cpu, True)

    def test_set_force_cpu_false(self):
        """
        Test for the set_force_cpu() method to set it to False
        """
        TorchUtils.set_force_cpu(False)
        self.assertEqual(TorchUtils.force_cpu, False)

    def test_set_force_cpu_none(self):
        """
        Test for the set_force_cpu() method to set it to None
        Assertion error is raised
        """
        with self.assertRaises(AssertionError):
            TorchUtils.set_force_cpu(None)

    def test_get_device(self):
        """
        Test for the get_device() method to get the accelerator type of the torch
        """
        TorchUtils.set_force_cpu(False)
        self.assertTrue(TorchUtils.get_device() == torch.device("cpu")
                        or TorchUtils.get_device() == torch.device("cuda"))

    def test_format_from_list(self):
        """
        Test to get a tensor from a list of data
        """
        data = [[1, 2]]
        tensor = TorchUtils.format(data,
                                   device=self.device,
                                   data_type=torch.float32)
        assert isinstance(tensor, torch.Tensor)

    def test_format_from_model(self):
        """
        Test to get a model into corresponding device
        """
        model = torch.nn.Linear(1, 1)
        model = TorchUtils.format(model)
        assert isinstance(model, torch.nn.Module)

    def test_format(self):
        """
        Test to format a tensor with a new data type
        """
        tensor = torch.randn(2, 2)
        tensor = TorchUtils.format(tensor,
                                   device=self.device,
                                   data_type=torch.float64)
        self.assertEqual(tensor.dtype, torch.float64)

    def test_format_none(self):
        """
        Test to format a tensor with none values for data type and device
        """
        device = None
        data_type = None
        tensor = torch.randn(2, 2)
        tensor = TorchUtils.format(tensor, device, data_type)
        self.assertEqual(tensor.dtype, torch.get_default_dtype())
        self.assertEqual(TorchUtils.get_device(), self.device)

    def test_format_from_numpy(self):
        """
        Test to get a torch tensor from a numpy array
        """
        data = [[1, 2], [3, 4]]
        numpy_data = np.array(data)
        tensor = TorchUtils.format(numpy_data, device=self.device)
        assert isinstance(tensor, torch.Tensor)

    def test_format_nn_module(self):
        """
        Test to format an instance of torch.nn.Module
        """
        data = torch.nn.Linear(20, 40)
        output = TorchUtils.format(data)
        assert isinstance(output, torch.nn.Module)

    def test_format_nn_module_force(self):
        """
        Test to format an instance of torch.nn.Module
        with force cpu set to False and with or without
        multiple cuda devices
        """
        TorchUtils.set_force_cpu(False)
        data = torch.nn.Linear(20, 40)
        output = TorchUtils.format(data)
        if torch.cuda.device_count() > 1:
            assert isinstance(output, torch.nn.DataParallel)
        else:
            assert isinstance(output, torch.nn.Module)

    def test_format_fail(self):
        """
        format function fails if a unsupported data type is provided
        """
        data = "Unsupported Data type"
        with self.assertRaises(TypeError):
            TorchUtils.format(data)

    def test_to_numpy(self):
        """
        Test to get a numpy array from a given torch tensor
        """
        data = torch.tensor([[1.0, -1.0], [1.0, -1.0]])
        tensor = TorchUtils.format(data, device=self.device)
        numpy_data = TorchUtils.to_numpy(tensor)
        assert isinstance(numpy_data, np.ndarray)
        tensor.requires_grad = True
        numpy_data = TorchUtils.to_numpy(tensor)
        assert isinstance(numpy_data, np.ndarray)

    def test_to_numpy_none(self):
        """
        Attribute error raised if the data type is not a torch tensor
        """
        data = None
        with self.assertRaises(AttributeError):
            TorchUtils.to_numpy(data)

        data = "invalid data type"
        with self.assertRaises(AttributeError):
            TorchUtils.to_numpy(data)

    def test_init_seed(self):
        """
        Test to initialize the seed and generating random values by resetting the seed to
        the same value everytime
        """
        TorchUtils.init_seed(0, deterministic=True)
        random1 = np.random.rand(4)
        TorchUtils.init_seed(0, deterministic=True)
        random2 = np.random.rand(4)
        self.assertEqual(random1[0], random2[0])

    def test_init_seed_none(self):
        """
        Test to generate a seed value by providing a none type
        """
        seed = TorchUtils.init_seed(None)
        assert isinstance(seed, int)

    def test_init_seed_fail(self):
        """
        Cannot generate a seed value if an unsupported type is provided
        """
        with self.assertRaises(TypeError):
            TorchUtils.init_seed("invalid type")


if __name__ == "__main__":
    unittest.main()
