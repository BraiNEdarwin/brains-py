import unittest

import torch
import numpy as np

import brainspy.processors.processor as processor
from brainspy.utils.pytorch import TorchUtils


class ProcessorTest(unittest.TestCase):
    """
    Class for testing 'processor.py'.
    """

    def test_merge_numpy(self):
        """
        Test merging numpy arrays.
        """
        inputs = np.array([
            [1.0, 5.0, 9.0, 13.0],
            [2.0, 6.0, 10.0, 14.0],
            [3.0, 7.0, 11.0, 15.0],
            [4.0, 8.0, 12.0, 16.0],
        ])
        control_voltages = inputs + np.ones(inputs.shape)
        input_indices = [0, 2, 4, 6]
        control_voltage_indices = [7, 5, 3, 1]
        result = processor.merge_electrode_data(
            inputs=inputs,
            control_voltages=control_voltages,
            input_indices=input_indices,
            control_voltage_indices=control_voltage_indices,
            use_torch=False,
        )
        self.assertEqual(result.shape, (4, 8))
        self.assertIsInstance(result, np.ndarray)
        target = np.array([
            [1.0, 14.0, 5.0, 10.0, 9.0, 6.0, 13.0, 2.0],
            [2.0, 15.0, 6.0, 11.0, 10.0, 7.0, 14.0, 3.0],
            [3.0, 16.0, 7.0, 12.0, 11.0, 8.0, 15.0, 4.0],
            [4.0, 17.0, 8.0, 13.0, 12.0, 9.0, 16.0, 5.0],
        ])
        for i in range(target.shape[0]):
            for j in range(target.shape[1]):
                self.assertEqual(result[i][j], target[i][j])

    def test_merge_torch(self):
        """
        Test merging torch tensors.
        """
        inputs = torch.tensor([
            [1.0, 5.0, 9.0, 13.0],
            [2.0, 6.0, 10.0, 14.0],
            [3.0, 7.0, 11.0, 15.0],
            [4.0, 8.0, 12.0, 16.0],
        ],
            device=TorchUtils.get_device(),
            dtype=TorchUtils.get_data_type())
        control_voltages = inputs + torch.ones(
            inputs.shape, dtype=TorchUtils.get_data_type())
        control_voltages.to(TorchUtils.get_device())
        input_indices = [0, 2, 4, 6]
        control_voltage_indices = [7, 5, 3, 1]
        result = processor.merge_electrode_data(
            inputs=inputs,
            control_voltages=control_voltages,
            input_indices=input_indices,
            control_voltage_indices=control_voltage_indices,
            use_torch=True,
        )
        self.assertEqual(result.shape, (4, 8))
        self.assertIsInstance(result, torch.Tensor)
        target = torch.tensor(
            [
                [1.0, 14.0, 5.0, 10.0, 9.0, 6.0, 13.0, 2.0],
                [2.0, 15.0, 6.0, 11.0, 10.0, 7.0, 14.0, 3.0],
                [3.0, 16.0, 7.0, 12.0, 11.0, 8.0, 15.0, 4.0],
                [4.0, 17.0, 8.0, 13.0, 12.0, 9.0, 16.0, 5.0],
            ],
            dtype=torch.float32,
        )
        for i in range(target.shape[0]):
            for j in range(target.shape[1]):
                self.assertEqual(result[i][j], target[i][j])


if __name__ == "__main__":
    unittest.main()
