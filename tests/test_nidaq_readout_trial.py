import unittest
import numpy as np
import random
import brainspy
from brainspy.processors.hardware.drivers.ni.setup import NationalInstrumentsSetup
from tests.test_utils import check_test_configs
from brainspy.processors.hardware.drivers.nidaq import CDAQtoNiDAQ


class NIDAQ_ReadoutTrial_Test(unittest.TestCase):
    """
    Test readout_trial() of the Nidaq Driver.

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
    def test_readout_trial_random(self):
        """
        Test the readout_trial with random input data and check
        if the readout is not none
        """
        configs = self.get_configs()
        a1 = random.randint(1, 1000)
        a2 = len(configs["instruments_setup"]["activation_channels"]) + 1
        nidaq = CDAQtoNiDAQ(configs)
        y = np.random.rand(a1, a2) / 1000
        # Force them to start and end at zero
        y[0] = np.zeros_like(a2)
        y[-1] = np.zeros_like(a2)
        try:
            nidaq.original_shape = y.shape[0]
            val = nidaq.readout_trial(y.T)
        except (Exception):
            self.fail("Could not synchronise output data")
        finally:
            nidaq.close_tasks()
            self.assertIsNotNone(val)

    @unittest.skipUnless(
        brainspy.__TEST_MODE__ == "HARDWARE_NIDAQ",
        "Method deactivated as it is only possible to be tested on a CDAQ TO NIDAQ setup"
    )
    def test_readout_trial_invalid_type(self):
        """
        Invalid type for input raises an AssertionError
        """
        configs = self.get_configs()
        nidaq = CDAQtoNiDAQ(configs)
        with self.assertRaises(AssertionError):
            nidaq.readout_trial("Invalid type")
        with self.assertRaises(AssertionError):
            nidaq.readout_trial(500)
        with self.assertRaises(AssertionError):
            nidaq.readout_trial(100.10)
        with self.assertRaises(AssertionError):
            nidaq.readout_trial({"dict_key": 2})
        with self.assertRaises(AssertionError):
            nidaq.readout_trial([1, 2, 3, 4, 5, 6])
        nidaq.close_tasks()


if __name__ == "__main__":
    testobj = NIDAQ_ReadoutTrial_Test()
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
