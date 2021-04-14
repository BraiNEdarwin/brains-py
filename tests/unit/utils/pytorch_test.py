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

    def __init__(self, test_name):
        super(PyTorchTest, self).__init__()

    def test_set_force_cpu(self):
        """
        Test for the set_force_cpu() method to set it to either True or False
        """
        TorchUtils.set_force_cpu(True)
        self.assertEqual(TorchUtils.force_cpu, True)

    def test_get_device(self):
        """
        Test for the get_device() method to get the accelerator type of the torch
        """
        TorchUtils.set_force_cpu(False)
        self.assertEqual(TorchUtils.get_device(), torch.device("cpu"))

    def test_format_from_list(self):
        """
        Test to get a tensor from a list of data
        """
        data = [[1, 2]]
        tensor = TorchUtils.format(data, data_type=torch.float32)
        assert isinstance(tensor, torch.Tensor)

    def test_format(self):
        """
        Test to format a tensor with a new data type
        """
        tensor = torch.randn(2, 2)
        tensor = TorchUtils.format(tensor, data_type=torch.float64)
        self.assertEqual(tensor.dtype, torch.float64)

    def test_format_from_numpy(self):
        """
        Test to get a torch tensor from a numpy array
        """
        data = [[1, 2], [3, 4]]
        numpy_data = np.array(data)
        tensor = TorchUtils.format(numpy_data)
        assert isinstance(tensor, torch.Tensor)

    def test_to_numpy(self):
        """
        Test to get a numpy array from a given torch tensor
        """
        tensor = torch.tensor([[1.0, -1.0], [1.0, -1.0]])
        numpy_data = TorchUtils.to_numpy(tensor)
        assert isinstance(numpy_data, np.ndarray)

    def test_init_seed(self):
        """
        Test to initialize the seed and generating random values by resetting the seed to the same value everytime
        """
        TorchUtils.init_seed(0)
        random1 = np.random.rand(4)
        TorchUtils.init_seed(0)
        random2 = np.random.rand(4)
        self.assertEqual(random1[0], random2[0])

    def runTest(self):

        self.test_set_force_cpu()
        self.test_get_device()
        self.test_format_from_list()
        self.test_format()
        self.test_format_from_numpy()
        self.test_to_numpy()
        self.test_init_seed()


if __name__ == "__main__":
    unittest.main()
