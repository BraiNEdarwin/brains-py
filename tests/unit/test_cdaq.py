import unittest
import numpy as np
import brainspy
from brainspy.processors.hardware.drivers.ni.setup import NationalInstrumentsSetup
from brainspy.processors.hardware.drivers.cdaq import CDAQtoCDAQ
from tests.unit.testing_utils import check_test_configs


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

    def get_configs_CDAQ(self):
        """
        Generate configurations to initialize the CDAQ driver
        """
        configs = {}
        configs["instrument_type"] = "cdaq_to_cdaq"
        configs["inverted_output"] = True
        configs["amplification"] = 100
        configs["output_clipping_range"] = [-1, 1]
        configs["instruments_setup"] = {}
        configs["instruments_setup"]["multiple_devices"] = False
        # TODO Specify the name of the Trigger Source
        configs["instruments_setup"]["trigger_source"] = "a"

        # TODO Specify the name of the Activation instrument
        configs["instruments_setup"]["activation_instrument"] = "b"

        # TODO Specify the Activation channels (pin numbers)
        # For example, [1,2,3,4,5,6,7]
        configs["instruments_setup"]["activation_channels"] = [
            1, 2, 3, 4, 5, 6, 7
        ]

        # TODO Specify the activation Voltage ranges
        # For example, [[-1.2, 0.6],[-1.2, 0.6],[-1.2, 0.6],[-1.2, 0.6],[-1.2, 0.6],[-0.7, 0.3],[-0.7, 0.3]]
        configs["instruments_setup"]["activation_voltage_ranges"] = [
            [-1.2, 0.6], [-1.2, 0.6], [-1.2, 0.6], [-1.2, 0.6], [-1.2, 0.6],
            [-0.7, 0.3], [-0.7, 0.3]
        ]

        # TODO Specify the name of the Readout Instrument
        configs["instruments_setup"]["readout_instrument"] = "c"

        # TODO Specify the readout channels
        # For example, [4]
        configs["instruments_setup"]["readout_channels"] = [4]
        configs["instruments_setup"]["activation_sampling_frequency"] = 500
        configs["instruments_setup"]["readout_sampling_frequency"] = 1000
        configs["instruments_setup"]["average_io_point_difference"] = True
        return configs

    @unittest.skipUnless(brainspy.TEST_MODE == "HARDWARE_CDAQ",
                         "Hardware test is skipped for simulation setup.")
    def test_init(self):
        """
        Test to check correct initialization of the Hardware processor with the cdaq_to_cdaq driver.
        """
        try:
            CDAQtoCDAQ(self.get_configs_CDAQ())
        except (Exception):
            self.fail("Could not initialize driver")

    def test_init_fail(self):
        """
        AssertionError is raised if the input is of an invalid type
        """
        with self.assertRaises(AssertionError):
            CDAQtoCDAQ([1, 2, 3, 4])
        with self.assertRaises(AssertionError):
            CDAQtoCDAQ("invalid type")
        with self.assertRaises(AssertionError):
            CDAQtoCDAQ(100)

    def test_init_keyerror(self):
        """
        KeyErrror is raised if some configs are missing
        """
        with self.assertRaises(KeyError):
            configs = self.get_configs_CDAQ()
            del configs["instruments_setup"]["activation_channels"]
            CDAQtoCDAQ(configs)

        with self.assertRaises(KeyError):
            configs = self.get_configs_CDAQ()
            del configs["instruments_setup"]["readout_channels"]
            CDAQtoCDAQ(configs)

        with self.assertRaises(KeyError):
            configs = self.get_configs_CDAQ()
            del configs["instruments_setup"]
            CDAQtoCDAQ(configs)

    @unittest.skipUnless(brainspy.TEST_MODE == "HARDWARE_CDAQ",
                         "Hardware test is skipped for simulation setup.")
    def test_forward_numpy(self):
        """
        Test if a forward pass through the processor returns a tensor of the
        right shape (the numpy version).
        """
        try:
            model = CDAQtoCDAQ(self.get_configs_CDAQ())
            x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
            x = model.forward_numpy(x)
        except (Exception):
            self.fail("Could not do a forward pass")
        else:
            self.assertEqual(list(x.shape), [1])

    @unittest.skipUnless(brainspy.TEST_MODE == "HARDWARE_CDAQ",
                         "Hardware test is skipped for simulation setup.")
    def test_forward_numpy_fail(self):
        """
        Test for the forward pass with invalid data types
        """
        model = CDAQtoCDAQ(self.get_configs_CDAQ())
        with self.assertRaises(AssertionError):
            model.forward_numpy("Wrong data type")
        with self.assertRaises(AssertionError):
            model.forward_numpy(100)
        with self.assertRaises(AssertionError):
            model.forward_numpy([1, 2, 3, 4])
        with self.assertRaises(AssertionError):
            model.forward_numpy({})


if __name__ == "__main__":

    testobj = CDAQ_Processor_Test()
    configs = testobj.get_configs_CDAQ()
    try:
        NationalInstrumentsSetup.type_check(configs)
        if check_test_configs(configs):
            raise unittest.SkipTest("Configs are missing. Skipping all tests.")
        else:
            unittest.main()
    except (Exception):
        print(Exception)
        raise unittest.SkipTest(
            "Configs not specified correctly. Skipping all tests.")
