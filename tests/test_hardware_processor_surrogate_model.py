# import unittest
# import numpy as np
# import warnings
# import random
# import torch
# from brainspy.utils.pytorch import TorchUtils
# from brainspy.processors.simulation.processor import SurrogateModel
# from brainspy.utils.waveform import WaveformManager
# from brainspy.processors.hardware.processor import HardwareProcessor

# class Hardware_Processor_SM_Test(unittest.TestCase):
#     """
#     Tests for the hardware processor in with a surrogate model
#     """
#     def get_processor_configs_and_surrogate_model(self):
#         """
#         Get a Surrogate model and some configs to initialize the hardware processor
#         """
#         configs = {}
#         configs["waveform"] = {}
#         configs["waveform"]["plateau_length"] = 10
#         configs["waveform"]["slope_length"] = 30

#         model_data = {}
#         model_data["info"] = {}
#         model_data["info"]["model_structure"] = {
#             "hidden_sizes": [90, 90, 90],
#             "D_in": 7,
#             "D_out": 1,
#             "activation": "relu",
#         }
#         surrogate_model = SurrogateModel(model_data["info"]["model_structure"])
#         return configs, surrogate_model

#     def test_init(self):
#         """
#         Test to check correct initialization of the Hardware processor in simulation debug model.
#         Hardware processor is initialized as an instance of the Surrogate Model.
#         """
#         configs, surrogate_model = self.get_processor_configs_and_surrogate_model(
#         )
#         try:
#             model = HardwareProcessor(
#                 surrogate_model,
#                 slope_length=configs["waveform"]["slope_length"],
#                 plateau_length=configs["waveform"]["plateau_length"],
#             )
#             model = TorchUtils.format(model)
#         except (Exception):
#             self.fail("Could not initialize processor")
#         else:
#             isinstance(model.driver, SurrogateModel)

#     def test_init_fail(self):
#         """
#         Invalid type for arguments raises an AssertionError
#         """
#         with self.assertRaises(AssertionError):
#             configs, surrogate_model = self.get_processor_configs_and_surrogate_model(
#             )
#             HardwareProcessor(surrogate_model, [1, 2, 3, 4], 100)
#         with self.assertRaises(AssertionError):
#             configs, surrogate_model = self.get_processor_configs_and_surrogate_model(
#             )
#             HardwareProcessor(surrogate_model, "invalid_type", 100)
#         with self.assertRaises(AssertionError):
#             configs, surrogate_model = self.get_processor_configs_and_surrogate_model(
#             )
#             HardwareProcessor(surrogate_model, 50.5, {})
#         with self.assertRaises(AssertionError):
#             configs, surrogate_model = self.get_processor_configs_and_surrogate_model(
#             )
#             HardwareProcessor(surrogate_model, None, 100)

#     def test_forward(self):
#         """
#         Test if a forward pass through the processor returns a tensor of the
#         right shape.
#         """
#         randomlist = []
#         for i in range(0, 3):
#             newlist = []
#             for j in range(0, 7):
#                 newlist.append(random.randint(1, 100))
#             randomlist.append(newlist)
#         data = TorchUtils.format(randomlist)
#         try:
#             configs, surrogate_model = self.get_processor_configs_and_surrogate_model(
#             )
#             model = HardwareProcessor(
#                 surrogate_model,
#                 slope_length=configs["waveform"]["slope_length"],
#                 plateau_length=configs["waveform"]["plateau_length"],
#             )
#             model = TorchUtils.format(model)
#             mgr = WaveformManager(configs["waveform"])
#             data_plateaus = TorchUtils.format(mgr.points_to_plateaus(data))
#             x = model.forward(data_plateaus)
#         except (Exception):
#             self.fail("Could not do a forward pass")
#         else:
#             self.assertEqual(list(x.shape), [30, 1])

#     def test_forward_fail(self):
#         """
#         RunTimeError is raised if an invalid shape combination is provided.
#         Error : invalid shape - RuntimeError: mat1 and mat2 shapes cannot be multiplied
#         """

#         with self.assertRaises(RuntimeError):
#             randomlist = [1, 2, 3, 4, 5]
#             data = TorchUtils.format(randomlist)
#             configs, surrogate_model = self.get_processor_configs_and_surrogate_model(
#             )
#             model = HardwareProcessor(
#                 surrogate_model,
#                 slope_length=configs["waveform"]["slope_length"],
#                 plateau_length=configs["waveform"]["plateau_length"],
#             )
#             mgr = WaveformManager(configs["waveform"])
#             data_plateaus = mgr.points_to_plateaus(data)
#             model.forward(data_plateaus)

#     def test_forward_invalid_type(self):
#         """
#         AssertionError is raised if an invalid type is provided to the forward function
#         """
#         configs, surrogate_model = self.get_processor_configs_and_surrogate_model(
#         )
#         model = HardwareProcessor(
#             surrogate_model,
#             slope_length=configs["waveform"]["slope_length"],
#             plateau_length=configs["waveform"]["plateau_length"],
#         )
#         with self.assertRaises(AssertionError):
#             model.forward([1, 2, 3, 4])
#         with self.assertRaises(AssertionError):
#             model.forward({})
#         with self.assertRaises(AssertionError):
#             model.forward(100)
#         with self.assertRaises(AssertionError):
#             model.forward("invalid type")

#     def test_forward_numpy(self):
#         """
#         Test if a forward pass through the processor returns a tensor of the
#         right shape (the numpy version).
#         """
#         try:
#             configs, surrogate_model = self.get_processor_configs_and_surrogate_model(
#             )
#             model = HardwareProcessor(
#                 surrogate_model,
#                 slope_length=configs["waveform"]["slope_length"],
#                 plateau_length=configs["waveform"]["plateau_length"],
#             )
#             model.driver = TorchUtils.format(model.driver)
#             randomlist = []
#             for i in range(0, 7):
#                 randomlist.append(random.randint(0, 100))
#             x = np.array(randomlist)
#             x = model.forward_numpy(x)
#         except (Exception):
#             self.fail("Could not do forward pass on this numpy data")
#         else:
#             self.assertEqual(list(x.shape), [1])

#     def test_forward_numpy_invalid_type(self):
#         """
#         AssertionError is raised if an invalid type is provided to the forward_numpy function
#         """
#         configs, surrogate_model = self.get_processor_configs_and_surrogate_model(
#         )
#         model = HardwareProcessor(
#             surrogate_model,
#             slope_length=configs["waveform"]["slope_length"],
#             plateau_length=configs["waveform"]["plateau_length"],
#         )
#         with self.assertRaises(AssertionError):
#             model.forward_numpy([1, 2, 3, 4])
#         with self.assertRaises(AssertionError):
#             model.forward_numpy({})
#         with self.assertRaises(AssertionError):
#             model.forward_numpy(100)
#         with self.assertRaises(AssertionError):
#             model.forward_numpy("invalid type")

#     def test_close(self):
#         """
#         Test if closing the processor raises a warning.
#         """
#         configs, surrogate_model = self.get_processor_configs_and_surrogate_model(
#         )
#         model = HardwareProcessor(
#             surrogate_model,
#             slope_length=configs["waveform"]["slope_length"],
#             plateau_length=configs["waveform"]["plateau_length"],
#         )
#         with warnings.catch_warnings(record=True) as caught_warnings:
#             warnings.simplefilter("always")
#             model.close()
#             self.assertEqual(len(caught_warnings), 1)

#     def test_is_hardware(self):
#         """
#         Test if the processor is a hardware,but in this case is an instance of a Surrogate Model.
#         """
#         configs, surrogate_model = self.get_processor_configs_and_surrogate_model(
#         )
#         model = HardwareProcessor(
#             surrogate_model,
#             slope_length=configs["waveform"]["slope_length"],
#             plateau_length=configs["waveform"]["plateau_length"],
#         )
#         self.assertFalse(model.is_hardware())

#     def test_get_voltage_ranges(self):
#         """
#         Test to get voltage ranges which returns a nonetype incase of a SurrogateModel
#         """
#         configs, surrogate_model = self.get_processor_configs_and_surrogate_model(
#         )
#         model = HardwareProcessor(
#             surrogate_model,
#             slope_length=configs["waveform"]["slope_length"],
#             plateau_length=configs["waveform"]["plateau_length"],
#         )
#         self.assertIsNone(
#             model.get_voltage_ranges())  # none only for surrogate model

#     def test_get_clipping_value(self):
#         """
#         Test to get the clipping value and assert it is an instance of a torch Tensor
#         """
#         configs, surrogate_model = self.get_processor_configs_and_surrogate_model(
#         )
#         model = HardwareProcessor(
#             surrogate_model,
#             slope_length=configs["waveform"]["slope_length"],
#             plateau_length=configs["waveform"]["plateau_length"],
#         )
#         self.assertIsNotNone(model.get_clipping_value())
#         self.assertIsInstance(model.get_clipping_value(), torch.Tensor)

# if __name__ == "__main__":
#     unittest.main()
