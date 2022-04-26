import unittest
import numpy as np
import brainspy
import math
from brainspy.processors.hardware.drivers.ni.tasks import IOTasksManager
from brainspy.processors.hardware.drivers.ni.setup import NationalInstrumentsSetup


class Setup_Test(unittest.TestCase):
    """
    Tests for the NationalInstrumentsSetup
    """

    def get_configs(self):
        """
        Get the sample configs for the NationalInstrumentsSetup
        """
        configs = {}
        configs["output_clipping_range"] = [-1, 1]
        configs["amplification"] = 50.5
        configs["instrument_type"] = "cdaq_to_nidaq"
        configs["inverted_output"] = True
        configs["max_ramping_time_seconds"] = 0.001
        configs["instruments_setup"] = {}
        configs["instruments_setup"]["activation_sampling_frequency"] = 1000
        configs["instruments_setup"]["readout_sampling_frequency"] = 2000
        configs["instruments_setup"]["multiple_devices"] = False
        configs["instruments_setup"]["trigger_source"] = "cDAQ1/segment1"
        configs["instruments_setup"]["activation_instrument"] = "cDAQ1Mod3"
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
        configs["instruments_setup"]["readout_instrument"] = "cDAQ1Mod4"
        configs["instruments_setup"]["readout_channels"] = [4]
        return configs

    def test_init(self):
        """
        Test to check correct initialization of the Setup with the cdaq_to_cdaq driver.
        """
        try:
            setup = NationalInstrumentsSetup(self.get_configs())
        except (Exception):
            self.fail("Could not initialize the NationalInstrumentsSetup")
        else:
            self.assertEqual(setup.inversion, -1)
            self.assertEqual(setup.last_points_to_write_val, -1)
            self.assertIsNone(setup.data_results)
            self.assertIsNone(setup.offsetted_points_to_write)
            self.assertIsNone(setup.timeout)
            self.assertIsInstance(setup.tasks_driver, IOTasksManager)

    def test_init_fail_output_clipping_range(self):

        configs = self.get_configs()
        with self.assertRaises(AssertionError):
            configs["output_clipping_range"] = {}
            NationalInstrumentsSetup(configs)
        with self.assertRaises(AssertionError):
            configs["output_clipping_range"] = [1, 2, 3, 4]
            NationalInstrumentsSetup(configs)
        with self.assertRaises(AssertionError):
            configs["output_clipping_range"] = ["invalid", 5]
            NationalInstrumentsSetup(configs)
        with self.assertRaises(AssertionError):
            configs["output_clipping_range"] = [5, [1, 2, 3, 4]]
            NationalInstrumentsSetup(configs)

    def test_init_fail_amplification(self):

        configs = self.get_configs()
        with self.assertRaises(AssertionError):
            configs["amplification"] = "invalid type"
            NationalInstrumentsSetup(configs)
        with self.assertRaises(AssertionError):
            configs["amplification"] = {}
            NationalInstrumentsSetup(configs)
        with self.assertRaises(AssertionError):
            configs["amplification"] = [1, 2, 34]
            NationalInstrumentsSetup(configs)

    def test_init_fail_devices(self):

        configs = self.get_configs()
        with self.assertRaises(AssertionError):
            configs["instruments_setup"][
                "activation_instrument"] = "invalid type"
            NationalInstrumentsSetup(configs)
        with self.assertRaises(AssertionError):
            configs["instruments_setup"]["readout_instrument"] = -5
            NationalInstrumentsSetup(configs)
        with self.assertRaises(AssertionError):
            configs["instruments_setup"]["trigger_source"] = 100
            NationalInstrumentsSetup(configs)
        with self.assertRaises(AssertionError):
            configs["instruments_setup"]["activation_sampling_frequency"] = {}
            NationalInstrumentsSetup(configs)
        with self.assertRaises(AssertionError):
            configs["instruments_setup"]["readout_sampling_frequency"] = [
                1, 2, 3, 4
            ]
            NationalInstrumentsSetup(configs)
        with self.assertRaises(AssertionError):
            configs["instruments_setup"]["readout_channels"] = "invalidtype"
            NationalInstrumentsSetup(configs)

    def test_init_activation_channels_error_1(self):
        """
        AssertionError is raised if the channel_names are not of type - list or numpy array
        """
        configs = self.get_configs()
        with self.assertRaises(AssertionError):
            configs["instruments_setup"]["activation_channels"] = {}
            NationalInstrumentsSetup(configs)

        configs = self.get_configs()
        with self.assertRaises(AssertionError):
            configs["instruments_setup"]["activation_channels"] = 100
            NationalInstrumentsSetup(configs)

        configs = self.get_configs()
        with self.assertRaises(AssertionError):
            configs["instruments_setup"][
                "activation_channels"] = "invalid type"
            NationalInstrumentsSetup(configs)

    def test_init_activation_channels_error_2(self):
        """
        AssertionError is raised if the voltage_ranges are not of type - list or numpy array
        """
        configs = self.get_configs()
        with self.assertRaises(AssertionError):
            configs["instruments_setup"]["activation_voltage_ranges"] = {}
            NationalInstrumentsSetup(configs)

        configs = self.get_configs()
        with self.assertRaises(AssertionError):
            configs["instruments_setup"]["activation_voltage_ranges"] = 100
            NationalInstrumentsSetup(configs)

        configs = self.get_configs()
        with self.assertRaises(AssertionError):
            configs["instruments_setup"][
                "activation_voltage_ranges"] = "invalid type"
            NationalInstrumentsSetup(configs)

    def test_init_activation_channels_error_3(self):
        """
        AssertionError is raised if the length of channel names is not equal to the
        length of voltage_ranges
        """
        configs = self.get_configs()
        configs["instruments_setup"]["activation_channels"] = [1, 2, 3, 4]
        configs["instruments_setup"]["activation_voltage_ranges"] = [
            [-1.2, 0.6],
            [-1.2, 0.6],
            [-1.2, 0.6],
            [-1.2, 0.6],
            [-1.2, 0.6],
            [-0.7, 0.3],
            [-0.7, 0.3],
        ]
        with self.assertRaises(AssertionError):
            NationalInstrumentsSetup(configs)

    def test_init_activation_channels_error_4(self):
        """
        AssertionError is raised if each voltage range is not a list or a numpy array
        """
        configs = self.get_configs()
        configs["instruments_setup"]["activation_voltage_ranges"] = [
            [-1.2, 0.6],
            [-1.2, 0.6],
            [-1.2, 0.6],
            [-1.2, 0.6],
            "Invalid type",
            [-0.7, 0.3],
            [-0.7, 0.3],
        ]
        with self.assertRaises(AssertionError):
            NationalInstrumentsSetup(configs)

    def test_init_activation_channels_error_5(self):
        """
        AssertionError is raised if each voltage range is not a list or a numpy array of length = 2
        """
        configs = self.get_configs()
        configs["instruments_setup"]["activation_voltage_ranges"] = [
            [-1.2, 0.6],
            [-1.2, 0.6],
            [-1.2, 0.6],
            [-1.2, 0.6],
            [-1.2, 0.6, 1.2],
            [-0.7, 0.3],
            [-0.7, 0.3],
        ]
        with self.assertRaises(AssertionError):
            NationalInstrumentsSetup(configs)

    def test_init_activation_channels_error_6(self):
        """
        AssertionError is raised if a voltage range does not contain an int/float type value
        """
        configs = self.get_configs()
        configs["instruments_setup"]["activation_voltage_ranges"] = [
            [-1.2, 0.6],
            [-1.2, 0.6],
            [-1.2, 0.6],
            [-1.2, 0.6],
            [-1.2, "Invalid type"],
            [-0.7, 0.3],
            [-0.7, 0.3],
        ]
        with self.assertRaises(AssertionError):
            NationalInstrumentsSetup(configs)

    def test_init_configs(self):

        try:
            setup = NationalInstrumentsSetup(self.get_configs())
            setup.init_configs()
        except (Exception):
            self.fail("Could not initialize the configs")
        else:
            self.assertEqual(setup.inversion, -1)
            self.assertEqual(setup.last_points_to_write_val, -1)
            self.assertIsNone(setup.data_results)
            self.assertIsNone(setup.offsetted_points_to_write)
            self.assertIsNone(setup.timeout)

    def test_sampling_configs(self):

        try:
            setup = NationalInstrumentsSetup(self.get_configs())
            setup.init_sampling_configs()
        except (Exception):
            self.fail("Could not initialize the sampling configs")
        else:
            self.assertIsNotNone(setup.io_point_difference)

    def test_sampling_configs_fail(self):

        configs = self.get_configs()
        configs["instruments_setup"]["activation_sampling_frequency"] = 3000
        configs["instruments_setup"]["readout_sampling_frequency"] = 2000
        with self.assertRaises(AssertionError):
            setup = NationalInstrumentsSetup()
            setup.init_sampling_configs()

    def test_init_tasks(self):
        try:
            setup = NationalInstrumentsSetup(self.get_configs())
            setup.init_tasks()
        except (Exception):
            self.fail("Could not initialize the tasks")
        else:
            self.assertIsInstance(setup.tasks_driver, IOTasksManager)

    def test_init_semaphore(self):
        try:
            setup = NationalInstrumentsSetup(self.get_configs())
            setup.init_semaphore()
        except (Exception):
            self.fail("Could not initialize the semaphore for the tasks")

    def test_enable_os_signals(self):
        try:
            setup = NationalInstrumentsSetup(self.get_configs())
            setup.enable_os_signals()
        except (Exception):
            self.fail("Could not enable_os_signals")

    def test_disable_os_signals(self):
        try:
            setup = NationalInstrumentsSetup(self.get_configs())
            setup.disable_os_signals()
        except (Exception):
            self.fail("Could not disable_os_signals")

    def test_os_signal_handler(self):
        try:
            setup = NationalInstrumentsSetup(self.get_configs())
            setup.os_signal_handler(signum=100)
        except (Exception):
            self.fail("Could not initialize the os_signal_handler")

    def test_get_amplification_value(self):
        try:
            setup = NationalInstrumentsSetup(self.get_configs())
            val = setup.get_amplification_value()
        except (Exception):
            self.fail("Could not get_amplification_value")
        else:
            self.assertEqual(val, 50.5)

    def test_read_security_checks(self):
        try:
            setup = NationalInstrumentsSetup(self.get_configs())
            y = 0  #TODO get correct val here
            #"number of inputs to the device" times "input points that you want to input to the device"
            setup.read_security_checks(y)
        except (Exception):
            self.fail("Could not read_security_checks")

    def test_close_tasks(self):
        try:
            setup = NationalInstrumentsSetup(self.get_configs())
            setup.close_tasks(signum=100)
        except (Exception):
            self.fail("Could not close_tasks")

    def test__read_data(self):
        try:
            setup = NationalInstrumentsSetup(self.get_configs())
            y = 0  # TODO (device_input_channel_no, data_point_no)
            val = setup._read_data(y)
        except (Exception):
            self.fail("Could not _read_data from device")
        else:
            self.assertIsNotNone(val)

    def test_is_hardware(self):
        try:
            setup = NationalInstrumentsSetup(self.get_configs())
            val = setup.is_hardware()
        except (Exception):
            self.fail("Could not check if is_hardware")
        else:
            self.assertTrue(type(val) == bool)

    def test_set_timeout(self):
        #timeout=None
        try:
            setup = NationalInstrumentsSetup(self.get_configs())
            setup.set_timeout()
        except (Exception):
            self.fail("Could not set timeout for device")
        else:
            timeout = setup.offsetted_points_to_write * setup.io_point_difference
            test_timeout = (math.ceil(timeout) + 10)
            self.assertEqual(setup.timeout, test_timeout)
        #timeout=100
        try:
            setup = NationalInstrumentsSetup(self.get_configs())
            setup.set_timeout(100)
        except (Exception):
            self.fail("Could not set timeout for device")
        else:
            self.assertEqual(setup.timeout, 100)

    def test_calculate_io_points(self):
        try:
            setup = NationalInstrumentsSetup(self.get_configs())
            setup.calculate_io_points(1000)
        except (Exception):
            self.fail("Could not calculate_io_points")
        else:
            self.assertIsNotNone(setup.offsetted_points_to_read)
            self.assertIsNotNone(setup.offsetted_points_to_write)

    def test_set_io_configs(self):
        try:
            setup = NationalInstrumentsSetup(self.get_configs())
            setup.set_io_configs(1000, 100)
        except (Exception):
            self.fail("Could not set_io_configs")

    def test_read_data(self):
        try:
            setup = NationalInstrumentsSetup(self.get_configs())
            y = 0  # TODO (device_input_channel_no, data_point_no)
            val = setup.read_data(y)
        except (Exception):
            self.fail("Could not read_data from device")
        else:
            self.assertIsNotNone(val)

    def test_average_point_difference(self):
        try:
            setup = NationalInstrumentsSetup(self.get_configs())
            y = 0  # TODO (channel_no, read_point_no)
            val = setup.average_point_difference(y)
        except (Exception):
            self.fail("Could not get average_point_difference")
        else:
            self.assertIsNotNone(val)

    def test_process_output_data(self):
        try:
            setup = NationalInstrumentsSetup(self.get_configs())
            data = 0  # TODO (channel_no, read_point_no)
            val = setup.process_output_data(data)
        except (Exception):
            self.fail("Could not get average_point_difference")
        else:
            self.assertIsNotNone(val)


if __name__ == "__main__":
    unittest.main()
