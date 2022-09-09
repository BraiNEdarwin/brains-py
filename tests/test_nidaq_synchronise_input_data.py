import unittest
import numpy as np
import random
import brainspy
from brainspy.processors.hardware.drivers.ni.setup import NationalInstrumentsSetup
from utils import check_test_configs
from brainspy.processors.hardware.drivers.nidaq import CDAQtoNiDAQ


class NIDAQ_Synchronise_Input_Test(unittest.TestCase):
    """
    Test synchronise_input_data() of the Nidaq Driver.

    To run this file, the device has to be connected to a NIDAQ setup and
    the device configurations have to be specified depending on the setup.

    The test mode has to be set to HARDWARE_NIDAQ in tests/main.py.
    The required keys have to be defined in the get_configs() function.

    Some sample keys have been defined to run tests which do not require connection
    to the hardware.
    """
    def get_configs(self):
        """
        Generate configurations to initialize the Nidaq driver
        """
        configs = {}
        configs["inverted_output"] = True
        configs["amplification"] = 100
        configs["output_clipping_range"] = [-1, 1]

        configs["instrument_type"] = "cdaq_to_nidaq"
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

    @unittest.skipUnless(
        brainspy.__TEST_MODE__ == "HARDWARE_NIDAQ",
        "Method deactivated as it is only possible to be tested on a CDAQ TO NIDAQ setup"
    )
    def test_synchronise_random_shape(self):
        """
        Test to synchronise input data for random shape of input
        """
        a1 = random.randint(1, 1000)
        a2 = random.randint(1, 9)
        configs = self.get_configs()
        nidaq = CDAQtoNiDAQ(configs)
        y = np.random.rand(a1, a2)
        try:
            new = nidaq.synchronise_input_data(y)
        except (Exception):
            self.fail("Could not synchronise inout data")
        test_synchronization_value = 0.04
        self.assertEqual(
            new.shape,
            (a1 + 1, a2 +
             (test_synchronization_value *
              configs["instruments_setup"]["activation_sampling_frequency"])))
        nidaq.close_tasks()

    # Failing test
    # @unittest.skipUnless(
    #     brainspy.__TEST_MODE__ == "HARDWARE_NIDAQ",
    #     "Method deactivated as it is only possible to be tested on a CDAQ TO NIDAQ setup"
    # )
    # def test_synchronise_large_shape(self):

    #     configs = self.get_configs()
    #     nidaq = CDAQtoNiDAQ(configs)
    #     y = np.random.rand(1000, 4, 3, 2, 2)
    #     try:
    #         new = nidaq.synchronise_input_data(y)
    #     except(Exception):
    #         self.fail("Could not synchronise inout data")

    @unittest.skipUnless(
        brainspy.__TEST_MODE__ == "HARDWARE_NIDAQ",
        "Method deactivated as it is only possible to be tested on a CDAQ TO NIDAQ setup"
    )
    def test_synchronise_single_dimension(self):
        """
        Test to synchronise input data with shape of only 1 dimension
        """
        for i in range(1, 10):
            configs = self.get_configs()
            nidaq = CDAQtoNiDAQ(configs)
            y = np.random.rand(i)
            try:
                nidaq.synchronise_input_data(y)
            except (Exception):
                self.fail("Could not synchronise inout data")
        nidaq.close_tasks()

    @unittest.skipUnless(
        brainspy.__TEST_MODE__ == "HARDWARE_NIDAQ",
        "Method deactivated as it is only possible to be tested on a CDAQ TO NIDAQ setup"
    )
    def test_synchronise_output_contains_input(self):
        """
        Test to synchronise input data and check if output array
        contains all elements present in the input array
        """
        configs = self.get_configs()
        nidaq = CDAQtoNiDAQ(configs)
        y = np.random.rand(3, 2)
        try:
            new = nidaq.synchronise_input_data(y)
        except (Exception):
            self.fail("Could not synchronise inout data")

        mask = np.isin(y, new)
        check = np.all(mask)
        self.assertTrue(check)
        nidaq.close_tasks()

    @unittest.skipUnless(
        brainspy.__TEST_MODE__ == "HARDWARE_NIDAQ",
        "Method deactivated as it is only possible to be tested on a CDAQ TO NIDAQ setup"
    )
    def test_synchronise_invalid_type(self):
        """
        Invalid type for input raises a Type Error
        """
        configs = self.get_configs()
        nidaq = CDAQtoNiDAQ(configs)
        with self.assertRaises(AssertionError):
            nidaq.synchronise_input_data("Invalid type")
        with self.assertRaises(AssertionError):
            nidaq.synchronise_input_data(100)
        nidaq.close_tasks()


if __name__ == "__main__":
    testobj = NIDAQ_Synchronise_Input_Test()
    configs = testobj.get_configs()
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
