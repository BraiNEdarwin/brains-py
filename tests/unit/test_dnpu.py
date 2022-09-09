import unittest

import torch
import copy
import numpy as np

import brainspy
from brainspy.processors import dnpu
from brainspy.processors.modules import bn
from brainspy.utils.pytorch import TorchUtils
from brainspy.processors.processor import Processor
from tests.unit.testing_utils import is_hardware_fake


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
        self.info['electrode_info']['activation_electrodes'][
            'electrode_no'] = 7
        self.info['electrode_info']['activation_electrodes'][
            'voltage_ranges'] = np.array([[-0.55, 0.325], [-0.95, 0.55],
                                          [-1., 0.6], [-1., 0.6], [-1., 0.6],
                                          [-0.95, 0.55], [-0.55, 0.325]])
        self.info['electrode_info']['output_electrodes'] = {}
        self.info['electrode_info']['output_electrodes']['electrode_no'] = 1
        self.info['electrode_info']['output_electrodes']['amplification'] = [
            28.5
        ]
        self.info['electrode_info']['output_electrodes'][
            'clipping_value'] = None

        self.data_input_indices = [[3, 4]]
        self.node = TorchUtils.format(Processor(self.configs, self.info))
        self.model = TorchUtils.format(
            dnpu.DNPU(self.node, data_input_indices=self.data_input_indices))

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

    def test_init(self):
        try:
            model = None
            model = TorchUtils.format(
                dnpu.DNPU(self.node,
                          data_input_indices=self.data_input_indices))
        except Exception:
            self.fail('Init with vec pass type failed')
        del model

        try:
            model = None
            model = TorchUtils.format(
                dnpu.DNPU(self.node,
                          data_input_indices=self.data_input_indices,
                          forward_pass_type='for'))
        except Exception:
            self.fail('Init with for pass type failed')
        del model

        with self.assertRaises(AssertionError):
            model = None
            model = TorchUtils.format(
                dnpu.DNPU(self.node, data_input_indices=[2, 3]))
        del model

        with self.assertRaises(AssertionError):
            model = None
            model = TorchUtils.format(
                dnpu.DNPU(self.node, data_input_indices=None))
        del model

        with self.assertRaises(AssertionError):
            model = None
            model = TorchUtils.format(
                dnpu.DNPU(self.node, data_input_indices=[]))
        del model

        with self.assertRaises(AssertionError):
            model = None
            model = TorchUtils.format(
                dnpu.DNPU(self.node, data_input_indices="str"))
            del model

        with self.assertRaises(AssertionError):
            model = None
            model = TorchUtils.format(dnpu.DNPU(None,
                                                data_input_indices="str"))
            del model

        with self.assertRaises(AssertionError):
            model = None
            model = TorchUtils.format(
                dnpu.DNPU("str", data_input_indices="str"))
            del model

        with self.assertRaises(AssertionError):
            model = None
            model = TorchUtils.format(
                dnpu.DNPU(self.node, data_input_indices=[[24, 18]]))
            del model

        with self.assertRaises(AssertionError):
            model = None
            model = TorchUtils.format(
                dnpu.DNPU(self.node,
                          data_input_indices=[[1, 2, 3, 4, 5, 6, 7, 8, 9,
                                               10]]))
            del model

        with self.assertRaises(AssertionError):
            model = None
            model = TorchUtils.format(
                dnpu.DNPU(self.node, data_input_indices=[[1, 2], [1, 1]]))
            del model

    def test_electrodes(self):
        in_electrode_no = self.model.get_data_input_electrode_no()
        control_electrode_no = self.model.get_control_electrode_no()
        self.assertTrue(
            (in_electrode_no + control_electrode_no
             ) == self.model.processor.get_activation_electrode_no())
        self.assertFalse(self.model.is_hardware())
        control_voltages = self.model.get_control_voltages()
        self.model.reset()
        control_voltages2 = self.model.get_control_voltages()
        self.assertFalse(
            torch.eq(control_voltages,
                     control_voltages2).all().detach().cpu().item())
        self.assertTrue(
            torch.eq(self.model.get_input_ranges(),
                     self.model.data_input_ranges).all().detach().cpu().item())
        self.assertEqual(self.model.get_info_dict(), self.model.processor.info)

    def test_regularizer(self):
        ver = torch.__version__
        torch.__version__ = '1.10.0'
        try:
            model = None
            model = TorchUtils.format(
                dnpu.DNPU(self.node,
                          data_input_indices=self.data_input_indices))
            control_voltages = model.get_control_voltages()
            model.set_control_voltages(control_voltages * 1000)
            model.regularizer()
            control_voltages_res1 = model.get_control_voltages()
            torch.__version__ = ver
            model.set_control_voltages(control_voltages * 1000)
            model.regularizer()
            control_voltages_res2 = model.get_control_voltages()
            self.assertTrue(
                torch.eq(control_voltages_res1,
                         control_voltages_res2).all().detach().cpu().item())
        except Exception:
            self.fail('Unable to constraint control voltages')
        torch.__version__ = ver

    def test_forward_for(self):
        ver = torch.__version__
        torch.__version__ = '1.10.0'
        try:
            model = None
            model = TorchUtils.format(
                dnpu.DNPU(self.node,
                          data_input_indices=[[0, 3]],
                          forward_pass_type='for'))
            model.add_input_transform([0, 1], strict=True)
            x = TorchUtils.format(
                torch.rand((torch.randint(10, 15, (1, 1)).item(),
                            len(self.data_input_indices[0]))))
            res1 = model(x)
            torch.__version__ = ver
            model.remove_input_transform()
            model.add_input_transform([0, 1], strict=True)
            res2 = model(x)
            model.remove_input_transform()
            self.assertTrue(torch.eq(res1, res2).all().detach().cpu().item())
        except Exception:
            self.fail('Init with for pass type failed')
        if model is not None:
            del model

        torch.__version__ = ver

    def test_unique_control_ranges_linear_transform(self):
        try:
            model = None
            model = TorchUtils.format(
                dnpu.DNPU(self.node,
                          data_input_indices=[[2, 3]],
                          forward_pass_type='for'))
            model.add_input_transform([0, 1], strict=True)
            x = TorchUtils.format(
                torch.rand((torch.randint(10, 15, (1, 1)).item(),
                            len(self.data_input_indices[0]))))
            res1 = model(x)

        except Exception:
            self.fail('Init with for pass type failed')
        if model is not None:
            del model

    def test_constraint_control_voltages(self):
        ver = torch.__version__
        torch.__version__ = '1.10.0'
        try:
            model = None
            model = TorchUtils.format(
                dnpu.DNPU(self.node,
                          data_input_indices=self.data_input_indices))
            control_voltages = model.get_control_voltages()
            model.set_control_voltages(control_voltages * 1000)
            model.constraint_control_voltages()
            control_voltages_res1 = model.get_control_voltages()
            torch.__version__ = ver
            model.set_control_voltages(control_voltages * 1000)
            model.constraint_control_voltages()
            control_voltages_res2 = model.get_control_voltages()
            self.assertTrue(
                torch.eq(control_voltages_res1,
                         control_voltages_res2).all().detach().cpu().item())
        except Exception:
            self.fail('Unable to constraint control voltages')
        torch.__version__ = ver

    def test_sample_controls(self):
        ver = torch.__version__
        torch.__version__ = '1.10.0'
        try:
            model = None
            model = TorchUtils.format(
                dnpu.DNPU(self.node,
                          data_input_indices=self.data_input_indices))
            TorchUtils.init_seed(0)
            controls1 = model.sample_controls()
            torch.__version__ = ver
            TorchUtils.init_seed(0)
            controls2 = model.sample_controls()
            self.assertTrue(
                torch.eq(controls1, controls2).all().detach().cpu().item())
        except Exception:
            self.fail('Unable to constraint control voltages')
        torch.__version__ = ver

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
        result = dnpu.merge_electrode_data(
            input_data=inputs,
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

    def test_forward_pass_for_and_vec(self):
        try:
            x = TorchUtils.format(
                torch.rand((torch.randint(10, 20, (1, 1)).item(),
                            len(self.data_input_indices[0]))))
            self.model.set_forward_pass("for")
            result_for = self.model(x)
            self.model.set_forward_pass("vec")
            result_vec = self.model(x)
            self.assertTrue(torch.eq(result_for, result_vec).all())
        except Exception:
            self.fail("Failed setting forward pass DNPU")

    def test_forward_pass_for_and_vec_multiple_dnpus(self):
        model = None
        input_indices = torch.randint(
            0, 7, (torch.randint(2, 20, (1, 1)).item(), 2)).detach().cpu()
        # Filter out values with duplicates
        input_indices = torch.stack([
            e for e in input_indices if len(e.unique()) == len(e)
        ]).detach().cpu().tolist()
        model = TorchUtils.format(
            dnpu.DNPU(self.node, data_input_indices=input_indices))
        try:
            x = TorchUtils.format(
                torch.rand((torch.randint(10, 15, (1, 1)).item(),
                            len(self.data_input_indices[0]))))
            self.model.set_forward_pass("for")
            result_for = self.model(x)
            self.model.set_forward_pass("vec")
            result_vec = self.model(x)
            self.assertTrue(torch.eq(result_for, result_vec).all())
        except Exception:
            self.fail("Failed setting forward pass DNPU")
        del model

    def test_forward_pass(self):
        try:
            self.model.set_forward_pass("for")
            self.model.set_forward_pass("vec")
        except Exception:
            self.fail("Failed setting forward pass DNPU")

        with self.assertRaises(AssertionError):
            self.model.set_forward_pass("matrix")

        with self.assertRaises(AssertionError):
            self.model.set_forward_pass(["vec"])

        with self.assertRaises(AssertionError):
            self.model.set_forward_pass(None)

    def test_get_node_no(self):
        try:
            self.model.get_node_no()
        except Exception:
            self.fail("Failed calculating DNPU node no")

    def test_activ_elec(self):
        try:
            input_data_electrode_no, control_electrode_no = self.model.init_activation_electrode_no(
            )
        except Exception:
            self.fail("Failed Initializing activation electrode DNPU")

    def test_init_elec_info(self):
        try:
            self.model.init_electrode_info([[0, 2]])
        except Exception:
            self.fail("Failed Initializing electrode info DNPU")

    def test_sample_control(self):
        try:
            control_voltage = self.model.sample_controls()
        except Exception:
            self.fail("Failed sampling control voltage")

    def __TEST_MODE__l_swap_sw(self):
        try:
            model = None

            model = TorchUtils.format(
                dnpu.DNPU(self.node,
                          data_input_indices=self.data_input_indices))
            state_dict = self.node.processor.model.state_dict()
            x = TorchUtils.format(
                torch.rand((torch.randint(10, 15, (1, 1)).item(),
                            len(self.data_input_indices[0]))))
            res1 = model(x)

            model.sw_train(self.configs,
                           self.info,
                           model_state_dict=state_dict)
            model = TorchUtils.format(model)
            res2 = model(x)

            self.assertTrue(torch.eq(res1, res2).all().detach().cpu().item())
        except Exception:
            self.fail("Failed setting forward pass DNPU in software")
        del model

    @unittest.skipUnless(brainspy.__TEST_MODE__ == "HARDWARE_CDAQ"
                         or brainspy.__TEST_MODE__ == "HARDWARE_NIDAQ",
                         "Hardware test is skipped for simulation setup.")
    def __TEST_MODE__l_swap_hw(self):
        try:
            model = None
            hw_configs = copy.deepcopy(self.configs)
            hw_configs['processor_type'] = 'simulation_debug'

            model = TorchUtils.format(
                dnpu.DNPU(self.node,
                          data_input_indices=self.data_input_indices))
            state_dict = self.node.processor.model.state_dict()
            x = TorchUtils.format(
                torch.rand((torch.randint(10, 15, (1, 1)).item(),
                            len(self.data_input_indices[0]))))
            res1 = model(x)

            model.processor.processor.voltage_ranges[
                0] = model.processor.processor.voltage_ranges[2]
            model.hw_eval(hw_configs,
                          data_input_indices=self.data_input_indices)
            model = TorchUtils.format(model)
            model.processor.processor.driver.is_hardware = is_hardware_fake
            res2 = model(x)

            model.sw_train(self.configs,
                           self.info,
                           model_state_dict=state_dict)
            model = TorchUtils.format(model)
            res3 = model(x)

            self.assertTrue(torch.eq(res1, res3).all().detach().cpu().item())
        except Exception:
            self.fail("Failed setting forward pass DNPU in hardware")
        del model

    def test_dnpu_batch_norm(self):
        try:
            model = None
            model = TorchUtils.format(
                bn.DNPUBatchNorm(self.node,
                                 data_input_indices=self.data_input_indices))
            x = TorchUtils.format(
                torch.rand((torch.randint(10, 15, (1, 1)).item(),
                            len(self.data_input_indices[0]))))
            model(x)
            model.get_logged_variables()
        except Exception:
            self.fail('Init with vec pass type failed')
        del model


if __name__ == "__main__":
    unittest.main()
