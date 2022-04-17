import unittest

import torch
import numpy as np

from brainspy.processors import dnpu
from brainspy.utils.pytorch import TorchUtils
from brainspy.processors.processor import Processor

class ProcessorTest(unittest.TestCase):
    """
    Class for testing 'dnpu.py'.
    """
    def __init__(self, *args, **kwargs) -> None:
        super(ProcessorTest, self).__init__(*args, **kwargs)
        self.configs = {}
        self.configs["processor_type"] = "simulation"
        self.configs["electrode_effects"] = {}
        self.configs["electrode_effects"]["clipping_value"] = None
        self.configs["driver"] = {}
        self.configs["waveform"] = {}
        self.configs["waveform"]["plateau_length"] = 1
        self.configs["waveform"]["slope_length"] = 0

        self.info = {}
        self.info['model_structure'] = {}
        self.info['model_structure']['hidden_sizes'] = [90, 90, 90, 90, 90]
        self.info['model_structure']['D_in'] = 7
        self.info['model_structure']['D_out'] = 1
        self.info['model_structure']['activation'] = 'relu'
        self.info['electrode_info'] = {}
        self.info['electrode_info']['electrode_no'] = 8
        self.info['electrode_info']['activation_electrodes'] = {}
        self.info['electrode_info']['activation_electrodes']['electrode_no'] = 7
        self.info['electrode_info']['activation_electrodes'][
            'voltage_ranges'] = np.array([[-0.55, 0.325], [-0.95, 0.55],
                                          [-1., 0.6], [-1., 0.6], [-1., 0.6],
                                          [-0.95, 0.55], [-0.55, 0.325]])
        self.info['electrode_info']['output_electrodes'] = {}
        self.info['electrode_info']['output_electrodes']['electrode_no'] = 1
        self.info['electrode_info']['output_electrodes']['amplification'] = [28.5]
        self.info['electrode_info']['output_electrodes']['clipping_value'] = None

        self.node = Processor(self.configs, self.info)
        self.model = dnpu.DNPU(self.node, data_input_indices=[[3, 4]])

    # def test_merge_numpy(self):
    #     """
    #     Test merging numpy arrays.
    #     """
    #     inputs = TorchUtils.format(
    #         np.array([
    #             [1.0, 5.0, 9.0, 13.0],
    #             [2.0, 6.0, 10.0, 14.0],
    #             [3.0, 7.0, 11.0, 15.0],
    #             [4.0, 8.0, 12.0, 16.0],
    #         ]))
    #     control_voltages = inputs + TorchUtils.format(np.ones(inputs.shape))
    #     input_indices = [0, 2, 4, 6]
    #     control_voltage_indices = [7, 5, 3, 1]
    #     result = merge_electrode_data(input_data=inputs,
    #                                   control_data=control_voltages,
    #                                   input_data_indices=input_indices,
    #                                   control_indices=control_voltage_indices)
    #     self.assertEqual(result.shape, (4, 8))
    #     self.assertIsInstance(result, np.ndarray)
    #     target = np.array([
    #         [1.0, 14.0, 5.0, 10.0, 9.0, 6.0, 13.0, 2.0],
    #         [2.0, 15.0, 6.0, 11.0, 10.0, 7.0, 14.0, 3.0],
    #         [3.0, 16.0, 7.0, 12.0, 11.0, 8.0, 15.0, 4.0],
    #         [4.0, 17.0, 8.0, 13.0, 12.0, 9.0, 16.0, 5.0],
    #     ])
    #     for i in range(target.shape[0]):
    #         for j in range(target.shape[1]):
    #             self.assertEqual(result[i][j], target[i][j])

    def test_merge_torch(self):
        """
        Test merging torch tensors.
        """
        inputs = TorchUtils.format(
            torch.tensor(
                [
                    [1.0, 5.0, 9.0, 13.0],
                    [2.0, 6.0, 10.0, 14.0],
                    [3.0, 7.0, 11.0, 15.0],
                    [4.0, 8.0, 12.0, 16.0],
                ],
                device=TorchUtils.get_device(),
                dtype=torch.get_default_dtype(),
            ))
        control_voltages = inputs + TorchUtils.format(
            torch.ones(inputs.shape, dtype=torch.get_default_dtype()))
        control_voltages.to(TorchUtils.get_device())
        input_indices = [0, 2, 4, 6]
        control_voltage_indices = [7, 5, 3, 1]
        result = dnpu.merge_electrode_data(input_data=inputs,
                                      control_data=control_voltages,
                                      input_data_indices=input_indices,
                                      control_indices=control_voltage_indices)
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

    def test_forward_pass(self):
        try:
            self.model.set_forward_pass("for")
            self.model.set_forward_pass("vec")
        except:
            self.fail("Failed setting forward pass DNPU")

        with self.assertRaises(ValueError):
            self.model.set_forward_pass("matrix")

        with self.assertRaises(ValueError):
            self.model.set_forward_pass(["vec"])

    def test_init_node_no(self):
        try:
            self.model.init_node_no()
        except:
            self.fail("Failed calculating nodes DNPU")

    def test_activ_elec(self):
        try:
            input_data_electrode_no, control_electrode_no = self.model.init_activation_electrode_no()
        except:
            self.fail("Failed Initializing activation electrode DNPU")

    def test_init_elec_info(self):
        try:
            self.model.init_electrode_info([[0, 2]])
        except:
            self.fail("Failed Initializing electrode info DNPU")

if __name__ == "__main__":
    unittest.main()
