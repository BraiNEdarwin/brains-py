import unittest
import numpy as np
import brainspy
from brainspy.processors.hardware.drivers.cdaq import CDAQtoCDAQ


class CDAQ_Processor_Test(unittest.TestCase):
    """
    Tests for the hardware processor with the CDAQ to CDAQ driver.
    """

    def get_configs_CDAQ(self):
        """
        Generate configurations to initialize the CDAQ driver
        Devices used - cDAQ3Mod1,cDAQ3Mod2
        """
        configs = {}
        configs["instrument_type"] = "cdaq_to_cdaq"
        configs["real_time_rack"] = False
        configs["inverted_output"] = True
        configs["amplification"] = 100
        configs["output_clipping_range"] = [-1, 1]
        configs["instruments_setup"] = {}
        configs["instruments_setup"]["multiple_devices"] = False
        configs["instruments_setup"]["trigger_source"] = "cDAQ3/segment1"
        configs["instruments_setup"]["activation_instrument"] = "cDAQ3Mod1"
        configs["instruments_setup"]["activation_channels"] = [
            0,
            2,
            5,
            3,
            4,
            6,
            1,
        ]
        configs["instruments_setup"]["activation_voltage_ranges"] = [
            [-1.2, 0.6],
            [-1.2, 0.6],
            [-1.2, 0.6],
            [-1.2, 0.6],
            [-1.2, 0.6],
            [-0.7, 0.3],
            [-0.7, 0.3],
        ]
        configs["instruments_setup"]["readout_instrument"] = "cDAQ3Mod2"
        configs["instruments_setup"]["readout_channels"] = [4]
        configs["instruments_setup"]["activation_sampling_frequency"] = 500
        configs["instruments_setup"]["readout_sampling_frequency"] = 1000
        configs["instruments_setup"]["average_io_point_difference"] = True
        return configs

    def test_init(self):
        """
        Test to check correct initialization of the Hardware processor with the cdaq_to_cdaq driver.
        """
        try:
            CDAQtoCDAQ(self.get_configs_CDAQ())
        except(Exception):
            self.fail("Could not initialize driver")

    def test_init_fail(self):

        with self.assertRaises(AssertionError):
            CDAQtoCDAQ([1, 2, 3, 4])
        with self.assertRaises(AssertionError):
            CDAQtoCDAQ("invalid type")
        with self.assertRaises(AssertionError):
            CDAQtoCDAQ(100)

    def test_init_keyerror(self):

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

    def test_forward_numpy(self):
        """
        Test if a forward pass through the processor returns a tensor of the
        right shape (the numpy version).
        """
        try:
            model = CDAQtoCDAQ(self.get_configs_CDAQ)
            x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
            x = model.forward_numpy(x)
        except(Exception):
            self.fail("Could not do a forward pass")
        else:
            self.assertEqual(list(x.shape), [1])

    def test_forward_numpy_fail(self):
        """
        Test for the forward pass with invalid data types
        """
        model = CDAQtoCDAQ(self.get_configs_CDAQ)
        with self.assertRaises(AssertionError):
            model.forward_numpy("Wrong data type")
        with self.assertRaises(AssertionError):
            model.forward_numpy(100)
        with self.assertRaises(AssertionError):
            model.forward_numpy([1, 2, 3, 4])
        with self.assertRaises(AssertionError):
            model.forward_numpy({})


if __name__ == "__main__":
    unittest.main()
