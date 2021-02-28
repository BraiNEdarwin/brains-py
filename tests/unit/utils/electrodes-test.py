import unittest
import brainspy.utils.electrodes as electrodes
import numpy as np
import torch


class ElectrodesTest(unittest.TestCase):
    def test_merge_numpy(self):
        # Test merging numpy arrays.
        inputs = np.array(
            [
                [1.0, 5.0, 9.0, 13.0],
                [2.0, 6.0, 10.0, 14.0],
                [3.0, 7.0, 11.0, 15.0],
                [4.0, 8.0, 12.0, 16.0],
            ]
        )
        control_voltages = inputs + np.ones(inputs.shape)
        input_indices = [0, 2, 4, 6]
        control_voltage_indices = [7, 5, 3, 1]
        result = electrodes.merge_electrode_data(
            inputs=inputs,
            control_voltages=control_voltages,
            input_indices=input_indices,
            control_voltage_indices=control_voltage_indices,
            use_torch=False,
        )
        self.assertEqual(result.shape, (4, 8))
        self.assertIsInstance(result, np.ndarray)
        target = np.array(
            [
                [1.0, 14.0, 5.0, 10.0, 9.0, 6.0, 13.0, 2.0],
                [2.0, 15.0, 6.0, 11.0, 10.0, 7.0, 14.0, 3.0],
                [3.0, 16.0, 7.0, 12.0, 11.0, 8.0, 15.0, 4.0],
                [4.0, 17.0, 8.0, 13.0, 12.0, 9.0, 16.0, 5.0],
            ]
        )
        for i in range(target.shape[0]):
            for j in range(target.shape[1]):
                self.assertEqual(result[i][j], target[i][j])

    def test_merge_torch(self):
        # Test merging torch tensors.
        inputs = torch.tensor(
            [
                [1.0, 5.0, 9.0, 13.0],
                [2.0, 6.0, 10.0, 14.0],
                [3.0, 7.0, 11.0, 15.0],
                [4.0, 8.0, 12.0, 16.0],
            ],
            dtype=torch.float32,
        )
        control_voltages = inputs + torch.ones(inputs.shape, dtype=torch.float32)
        input_indices = [0, 2, 4, 6]
        control_voltage_indices = [7, 5, 3, 1]
        result = electrodes.merge_electrode_data(
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

    def test_line(self):
        # Test scale and offset, and evaluation at a point, of a line.
        x_min = 1
        y_min = 1
        x_max = 2
        y_max = 0
        # This is the line y = 2 - x.
        x_val = 3
        offset = electrodes.get_offset(
            y_min=y_min, y_max=y_max, x_min=x_min, x_max=x_max
        )
        scale = electrodes.get_scale(y_min=y_min, y_max=y_max, x_min=x_min, x_max=x_max)
        self.assertEqual(offset, 2)
        self.assertEqual(scale, -1)
        value = electrodes.transform_to_voltage(
            x_val=x_val, y_min=y_min, y_max=y_max, x_min=x_min, x_max=x_max
        )
        self.assertEqual(value, -1)

    def test_line_extreme(self):
        # Test scale and offset, and evaluation at a point, of a line.
        # Extreme case: x_min is larger than x_max.
        x_min = 2
        y_min = 0
        x_max = 1
        y_max = 1
        # This is the line y = 2 - x.
        x_val = 3
        offset = electrodes.get_offset(
            y_min=y_min, y_max=y_max, x_min=x_min, x_max=x_max
        )
        scale = electrodes.get_scale(y_min=y_min, y_max=y_max, x_min=x_min, x_max=x_max)
        self.assertEqual(offset, 2)
        self.assertEqual(scale, -1)
        value = electrodes.transform_to_voltage(
            x_val=x_val, y_min=y_min, y_max=y_max, x_min=x_min, x_max=x_max
        )
        self.assertEqual(value, -1)

    def test_line_multi_dim(self):
        # Test scale and offset, and evaluation at a point, of a line.
        # Using multi-dimensional data.
        x_min = np.array([1, 0, 0])
        y_min = np.array([1, 0, 1])
        x_max = np.array([2, 1, 1])
        y_max = np.array([0, 1, 1])
        # Lines are y = 2 - x; y = x; y = 1.
        x_val = np.array([3, 3, 3])
        offset = electrodes.get_offset(
            y_min=y_min, y_max=y_max, x_min=x_min, x_max=x_max
        )
        scale = electrodes.get_scale(y_min=y_min, y_max=y_max, x_min=x_min, x_max=x_max)
        value = electrodes.transform_to_voltage(
            x_val=x_val, y_min=y_min, y_max=y_max, x_min=x_min, x_max=x_max
        )
        self.assertTrue(np.array_equal(offset, np.array([2, 0, 1])))
        self.assertTrue(np.array_equal(scale, np.array([-1, 1, 0])))
        self.assertTrue(np.array_equal(value, np.array([-1, 3, 1])))


if __name__ == "__main__":
    unittest.main()
