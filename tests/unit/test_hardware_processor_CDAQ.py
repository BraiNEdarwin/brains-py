import unittest
import numpy as np
import warnings
import random
import torch
import brainspy
from tests.unit.testing_utils import check_test_configs
from brainspy.processors.hardware.drivers.cdaq import CDAQtoCDAQ
from brainspy.utils.pytorch import TorchUtils
from brainspy.utils.waveform import WaveformManager
from brainspy.processors.hardware.processor import HardwareProcessor
from brainspy.processors.hardware.drivers.ni.setup import NationalInstrumentsSetup
from tests.unit.testing_utils import get_configs

class Hardware_Processor_Test_CDAQ(unittest.TestCase):
    """
    Tests for the hardware processor with a CDAQ driver.

    To run this file, the device has to be connected to a CDAQ setup and
    the device configurations have to be specified depending on the setup.

    The test mode has to be set to HARDWARE_CDAQ in tests/main.py.
    The required keys have to be defined in the get_configs_CDAQ() function.

    Some sample keys have been defined to run tests which do not require connection
    to the hardware.

    """

    # def get_processor_configs(self):
    #     """
    #     Get the configs to initialize the hardware processor
    #     """
    #     configs = {}
    #     configs["waveform"] = {}
    #     configs["waveform"]["plateau_length"] = 10
    #     configs["waveform"]["slope_length"] = 30

    #     configs["amplification"] = 100
    #     configs["inverted_output"] = True
    #     configs["output_clipping_range"] = [-1, 1]

    #     configs["instrument_type"] = "cdaq_to_cdaq"

    #     configs["instruments_setup"] = {}

    #     configs["instruments_setup"]["multiple_devices"] = False
    #     # TODO Specify the name of the Trigger Source
    #     configs["instruments_setup"]["trigger_source"] = "a"

    #     # TODO Specify the name of the Activation instrument
    #     configs["instruments_setup"]["activation_instrument"] = "b"

    #     # TODO Specify the Activation channels (pin numbers)
    #     # For example, [1,2,3,4,5,6,7]
    #     configs["instruments_setup"]["activation_channels"] = [
    #         1, 2, 3, 4, 5, 6, 7
    #     ]

    #     # TODO Specify the activation Voltage ranges
    #     # For example, [[-1.2, 0.6],[-1.2, 0.6],[-1.2, 0.6],[-1.2, 0.6],[-1.2, 0.6],[-0.7, 0.3],[-0.7, 0.3]]
    #     configs["instruments_setup"]["activation_voltage_ranges"] = [
    #         [-1.2, 0.6], [-1.2, 0.6], [-1.2, 0.6], [-1.2, 0.6], [-1.2, 0.6],
    #         [-0.7, 0.3], [-0.7, 0.3]
    #     ]

    #     # TODO Specify the name of the Readout Instrument
    #     configs["instruments_setup"]["readout_instrument"] = "c"

    #     # TODO Specify the readout channels
    #     # For example, [4]
    #     configs["instruments_setup"]["readout_channels"] = [4]
    #     configs["instruments_setup"]["activation_sampling_frequency"] = 500
    #     configs["instruments_setup"]["readout_sampling_frequency"] = 1000
    #     configs["instruments_setup"]["average_io_point_difference"] = True

    #     return configs

    @unittest.skipUnless(brainspy.TEST_MODE == "HARDWARE_CDAQ",
                         "Hardware test is skipped for simulation setup.")
    def test_init(self):
        """
        Test to check correct initialization of the Hardware processor.
        """
        configs = get_configs()
        try:
            model = None
            model = HardwareProcessor(
                configs,
                slope_length=configs["waveform"]["slope_length"],
                plateau_length=configs["waveform"]["plateau_length"],
            )
        except (Exception):
            if model is not None:
                model.close()
            self.fail("Could not initialize processor")
        else:
            isinstance(model.driver, CDAQtoCDAQ)
            if model is not None:
                model.close()

    def test_init_fail(self):

        configs = get_configs()
        model = None
        with self.assertRaises(AssertionError):
            model = HardwareProcessor(configs, [1, 2, 3, 4], 100)
        if model is not None:
            model.close()

        model = None
        with self.assertRaises(AssertionError):
            HardwareProcessor(configs, "invalid_type", 100)
        if model is not None:
            model.close()

        model = None
        with self.assertRaises(AssertionError):
            HardwareProcessor(configs, 50.5, {})
        if model is not None:
            model.close()

        model = None
        with self.assertRaises(AssertionError):
            HardwareProcessor(configs, None, 100)
        if model is not None:
            model.close()

    @unittest.skipUnless(brainspy.TEST_MODE == "HARDWARE_CDAQ",
                         "Hardware test is skipped for simulation setup.")
    def test_forward(self):
        """
        Test if a forward pass through the processor returns a tensor of the
        right shape.
        """
        configs = get_configs()
        data = TorchUtils.format(torch.rand(np.random.randint(10,100),len(configs['instruments_setup']['activation_channels'])))
        data /= 50
        try:
            model = None
            
            model = HardwareProcessor(
                configs,
                slope_length=configs["waveform"]["slope_length"],
                plateau_length=configs["waveform"]["plateau_length"],
            )
            mgr = WaveformManager(configs["waveform"])
            data_plateaus = mgr.points_to_plateaus(data)
            x = model.forward(data_plateaus)
            x = mgr.plateaus_to_points(x)
        except (Exception):
            if model is not None:
                model.close()
            self.fail("Could not do a forward pass")
        else:
            self.assertEqual(x.shape[0], data.shape[0])
            if model is not None:
                model.close()

    @unittest.skipUnless(brainspy.TEST_MODE == "HARDWARE_CDAQ",
                         "Hardware test is skipped for simulation setup.")
    def test_forward_fail(self):
        """
        RunTimeError is raised if an invalid shape combination is provided.
        Error : invalid shape - RuntimeError: mat1 and mat2 shapes cannot be multiplied
        """

        with self.assertRaises(AssertionError):
            randomlist = [1, 2, 3, 4, 5]
            data = TorchUtils.format(randomlist)
            configs = get_configs()
            model = None
            model = HardwareProcessor(
                configs,
                slope_length=configs["waveform"]["slope_length"],
                plateau_length=configs["waveform"]["plateau_length"],
            )
            mgr = WaveformManager(configs["waveform"])
            data_plateaus = mgr.points_to_plateaus(data)
            model.forward(data_plateaus)
        if model is not None:
            model.close()

    @unittest.skipUnless(brainspy.TEST_MODE == "HARDWARE_CDAQ",
                         "Hardware test is skipped for simulation setup.")
    def test_forward_invalid_type(self):
        """
        AssertionError is raised if an invalid type is provided to the forward function
        """
        configs = get_configs()
        model = None
        model = HardwareProcessor(
            configs,
            slope_length=configs["waveform"]["slope_length"],
            plateau_length=configs["waveform"]["plateau_length"],
        )
        with self.assertRaises(AssertionError):
            model.forward([1, 2, 3, 4])
        with self.assertRaises(AssertionError):
            model.forward({})
        with self.assertRaises(AssertionError):
            model.forward(100)
        with self.assertRaises(AssertionError):
            model.forward("invalid type")
        if model is not None:
            model.close()

    # @unittest.skipUnless(brainspy.TEST_MODE == "HARDWARE_CDAQ",
    #                      "Hardware test is skipped for simulation setup.")
    # def test_forward_numpy(self):
    #     """
    #     Test if a forward pass through the processor returns a tensor of the
    #     right shape (the numpy version).
    #     """
    #     try:
    #         configs = get_configs()
    #         model = None
    #         model = HardwareProcessor(
    #             configs,
    #             slope_length=configs["waveform"]["slope_length"],
    #             plateau_length=configs["waveform"]["plateau_length"],
    #         )
    #         randomlist = []
    #         for i in range(0, 7):
    #             randomlist.append(random.randint(0, 100))
    #         x = np.array(randomlist)
    #         x = model.forward_numpy(x)
    #     except (Exception):
    #         if model is not None:
    #             model.close()
    #         self.fail("Could not do forward pass on this numpy data")
    #     else:
    #         self.assertEqual(list(x.shape), [1])
    #         if model is not None:
    #             model.close()

    @unittest.skipUnless(brainspy.TEST_MODE == "HARDWARE_CDAQ",
                         "Hardware test is skipped for simulation setup.")
    def test_forward_numpy_invalid_type(self):
        """
        AssertionError is raised if an invalid type is provided to the forward_numpy function
        """
        configs = get_configs()
        model = None
        model = HardwareProcessor(
            configs,
            slope_length=configs["waveform"]["slope_length"],
            plateau_length=configs["waveform"]["plateau_length"],
        )
        with self.assertRaises(AssertionError):
            model.forward_numpy([1, 2, 3, 4])
        with self.assertRaises(AssertionError):
            model.forward_numpy({})
        with self.assertRaises(AssertionError):
            model.forward_numpy(100)
        with self.assertRaises(AssertionError):
            model.forward_numpy("invalid type")
        if model is not None:
            model.close()

    @unittest.skipUnless(brainspy.TEST_MODE == "HARDWARE_CDAQ",
                         "Hardware test is skipped for simulation setup.")
    def test_close(self):
        """
        Test if closing the processor raises a warning.
        """
        configs = get_configs()
        model = HardwareProcessor(
            configs,
            slope_length=configs["waveform"]["slope_length"],
            plateau_length=configs["waveform"]["plateau_length"],
        )
        model.close()

    @unittest.skipUnless(brainspy.TEST_MODE == "HARDWARE_CDAQ",
                         "Hardware test is skipped for simulation setup.")
    def test_is_hardware(self):
        """
        Test if the processor is a hardware,but in this case is an instance of a Surrogate Model.
        """
        configs = get_configs()
        model = None
        model = HardwareProcessor(
            configs,
            slope_length=configs["waveform"]["slope_length"],
            plateau_length=configs["waveform"]["plateau_length"],
        )
        self.assertTrue(model.is_hardware())
        if model is not None:
            model.close()

    @unittest.skipUnless(brainspy.TEST_MODE == "HARDWARE_CDAQ",
                         "Hardware test is skipped for simulation setup.")
    def test_get_voltage_ranges(self):
        """
        Test to get voltage ranges which returns a nonetype incase of a SurrogateModel
        """
        configs = get_configs()
        model = None
        model = HardwareProcessor(
            configs,
            slope_length=configs["waveform"]["slope_length"],
            plateau_length=configs["waveform"]["plateau_length"],
        )
        self.assertIsNotNone(model.get_voltage_ranges())
        if model is not None:
            model.close()

    @unittest.skipUnless(brainspy.TEST_MODE == "HARDWARE_CDAQ",
                         "Hardware test is skipped for simulation setup.")
    def test_get_clipping_value(self):
        """
        Test to get the clipping value and assert it is an instance of a torch Tensor
        """
        configs = get_configs()
        model = None
        model = HardwareProcessor(
            configs,
            slope_length=configs["waveform"]["slope_length"],
            plateau_length=configs["waveform"]["plateau_length"],
        )
        try:    
            model.get_clipping_value()
        except Exception:
            if model is not None:
                model.close()
            self.fail('Unable to get clipping value')
        if model is not None:
            model.close()


if __name__ == "__main__":
    unittest.main()
