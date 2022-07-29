import unittest
import numpy as np
import brainspy
from brainspy.processors.hardware.drivers.ni.setup import NationalInstrumentsSetup
from brainspy.processors.hardware.drivers.cdaq import CDAQtoCDAQ
from tests.unit.testing_utils import get_configs


class CDAQ_Processor_Test(unittest.TestCase):
    """
    Tests for the CDAQ driver.

    To run this file, the device has to be connected to a CDAQ setup and
    the device configurations have to be specified depending on the setup.

    The test mode has to be set to HARDWARE_CDAQ in tests/main.py.
    The required keys have to be defined in the get_configs_CDAQ() function.

    Some sample keys have been defined to run tests which do not require connection
    to the hardware.
    """

    # def get_configs_CDAQ(self):
    #     """
    #     Generate configurations to initialize the CDAQ driver
    #     """
    #     configs = {}
    #     configs["instrument_type"] = "cdaq_to_cdaq"
    #     configs["inverted_output"] = True
    #     configs["amplification"] = [100]
    #     configs["output_clipping_range"] = [-1, 1]
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
    #     configs["instruments_setup"]["activation_channel_mask"] = [
    #         1, 1, 1, 1, 1, 1, 1
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
        Test to check correct initialization of the Hardware processor with the cdaq_to_cdaq driver.
        """
        model = None
        try:
            model = CDAQtoCDAQ(get_configs())
        except (Exception):
            if model is not None:
                model.close_tasks()
            self.fail("Could not initialize driver")
        else:
            if model is not None:
                model.close_tasks()

    @unittest.skipUnless(brainspy.TEST_MODE == "HARDWARE_CDAQ",
                         "Hardware test is skipped for simulation setup.")
    def test_init_fail(self):
        """
        AssertionError is raised if the input is of an invalid type
        """
        model = None
        with self.assertRaises(AssertionError):
            model = CDAQtoCDAQ([1, 2, 3, 4])
        if model is not None:
            model.close_tasks()
        model = None
        with self.assertRaises(AssertionError):
            model = CDAQtoCDAQ("invalid type")
        if model is not None:
            model.close_tasks()
        model = None
        with self.assertRaises(AssertionError):
            model = CDAQtoCDAQ(100)
        if model is not None:
            model.close_tasks()

    @unittest.skipUnless(brainspy.TEST_MODE == "HARDWARE_CDAQ",
                         "Hardware test is skipped for simulation setup.")
    def test_init_keyerror(self):
        """
        KeyErrror is raised if some configs are missing
        """
        model = None
        with self.assertRaises(AssertionError):
            configs = get_configs()
            del configs["instruments_setup"]["activation_channels"]
            model = CDAQtoCDAQ(configs)
        if model is not None:
            model.close_tasks()

        model = None
        with self.assertRaises(AssertionError):
            configs = get_configs()
            del configs["instruments_setup"]["readout_channels"]
            model = CDAQtoCDAQ(configs)
        if model is not None:
            model.close_tasks()

        with self.assertRaises(AssertionError):
            configs = get_configs()
            del configs["instruments_setup"]
            CDAQtoCDAQ(configs)

    @unittest.skipUnless(brainspy.TEST_MODE == "HARDWARE_CDAQ",
                         "Hardware test is skipped for simulation setup.")
    def test_forward_numpy(self):
        """
        Test if a forward pass through the processor returns a tensor of the
        right shape (the numpy version).
        """
        model = None
        x_dim = np.random.randint(2,10)
        try:
            model = CDAQtoCDAQ(get_configs())
            x = np.zeros((7,x_dim)).T #/ 1000#np.array([[0.1, 0.2, 0.1, 0.2, 0.13, 0.1, 0.2]]).T
            x = model.forward_numpy(x)

        except (Exception):
            if model is not None:
                model.close_tasks()
            self.fail("Could not do a forward pass")
        else:
            self.assertEqual(list(x.shape), [x_dim, 1])
            if model is not None:
                model.close_tasks()

    @unittest.skipUnless(brainspy.TEST_MODE == "HARDWARE_CDAQ",
                         "Hardware test is skipped for simulation setup.")
    def test_forward_numpy_fail(self):
        """
        Test for the forward pass with invalid data types
        """
        model = None
        model = CDAQtoCDAQ(get_configs())
        with self.assertRaises(AssertionError):
            model.forward_numpy("Wrong data type")
        if model is not None:
                model.close_tasks()

        model = None
        model = CDAQtoCDAQ(get_configs())
        with self.assertRaises(AssertionError):
            model.forward_numpy(100)
        if model is not None:
                model.close_tasks()

        model = None
        model = CDAQtoCDAQ(get_configs())
        with self.assertRaises(AssertionError):
            model.forward_numpy([1, 2, 3, 4])
        if model is not None:
                model.close_tasks()

        model = None
        model = CDAQtoCDAQ(get_configs())
        with self.assertRaises(AssertionError):
            model.forward_numpy({})
        if model is not None:
                model.close_tasks()


if __name__ == "__main__":
    unittest.main()
