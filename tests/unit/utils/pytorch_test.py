
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

    def test_set_data_type(self):
        """
        Test for the set_data_type() method to set it to a new data type
        """
        TorchUtils.set_data_type(torch.float64)
        self.assertEqual(TorchUtils.get_data_type(), torch.float64)

    def test_get_accelerator_type(self):
        """
        Test for the get_accelerator_type() method to get the accelerator type of the torch
        """
        TorchUtils.set_force_cpu(False)
        self.assertEqual(TorchUtils.get_accelerator_type(), torch.device("cpu"))

    def test_get_tensor_from_list(self):
        """
        Test to get a tensor from a list of data
        """
        data = [[1, 2]]
        tensor = TorchUtils.get_tensor_from_list(data, data_type=torch.float32)
        assert isinstance(tensor, torch.Tensor)

    def test_format_tensor(self):
        """
        Test to format a tensor with a new data type
        """
        tensor = torch.randn(2, 2)
        tensor = TorchUtils.format_tensor(tensor, data_type=torch.float64)
        self.assertEqual(tensor.dtype, torch.float64)

    def test_get_tensor_from_numpy(self):
        """
        Test to get a torch tensor from a numpy array
        """
        data = [[1, 2], [3, 4]]
        numpy_data = np.array(data)
        tensor = TorchUtils.get_tensor_from_numpy(numpy_data)
        assert isinstance(tensor, torch.Tensor)

    def test_get_numpy_from_tensor(self):
        """
        Test to get a numpy array from a given torch tensor
        """
        tensor = torch.tensor([[1., -1.], [1., -1.]])
        numpy_data = TorchUtils.get_numpy_from_tensor(tensor)
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
        self.test_set_data_type()
        self.test_get_accelerator_type()
        self.test_get_tensor_from_list()
        self.test_format_tensor()
        self.test_get_tensor_from_numpy()
        self.test_get_numpy_from_tensor()
        self.test_init_seed()


if __name__ == "__main__":
    unittest.main()
