import unittest
import numpy as np
import nidaqmx
import math
import random
import brainspy
from tests.unit.testing_utils import check_test_configs
from brainspy.processors.hardware.drivers.ni.tasks import IOTasksManager
from brainspy.processors.hardware.drivers.ni.setup import NationalInstrumentsSetup


class Setup_Test(unittest.TestCase):
    """
    Tests for the NationalInstrumentsSetup.

    To run this file, the device has to be connected to a CDAQ or a NIDAQ setup and
    the device configurations have to be specified depending on the setup.

    The test mode has to be set to HARDWARE_CDAQ or HARDWARE_NIDAQ in tests/main.py.
    The required keys have to be defined in the get_configs() function.

    Some sample keys have been defined to run tests which do not require connection
    to the hardware.
    """

    def get_configs(self):
        """
        Get the sample configs for the NationalInstrumentsSetup
        """
        configs = {}
        configs["output_clipping_range"] = [-1, 1]
        configs["amplification"] = 50.5

        # TODO Specify Instrument type
        # For a CDAQ setup, cdaq_to_cdaq.
        # For a NIDAQ setup, cdaq_to_nidaq.
        configs["instrument_type"] = "cdaq_to_cdaq"

        configs["inverted_output"] = True
        configs["max_ramping_time_seconds"] = 0.001
        configs["instruments_setup"] = {}
        configs["instruments_setup"]["activation_sampling_frequency"] = 1000
        configs["instruments_setup"]["readout_sampling_frequency"] = 2000
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

        return configs

    @unittest.skipUnless(brainspy.TEST_MODE == "HARDWARE_CDAQ"
                         or brainspy.TEST_MODE == "HARDWARE_NIDAQ",
                         "Hardware test is skipped for simulation setup.")
    def test_init(self):
        """
        Test to check correct initialization of the Setup
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
        """
        AssertionError is raised if the output clipping range is of an invalid type
        """
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
        """
        AssertionError is raised if the amplification value is of an invalid type
        """
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

    @unittest.skipUnless(brainspy.TEST_MODE == "HARDWARE_CDAQ"
                         or brainspy.TEST_MODE == "HARDWARE_NIDAQ",
                         "Hardware test is skipped for simulation setup.")
    def test_init_fail_devices(self):
        """
        nidaqmx.errors.DaqError is raised if the device names is incorrect
        """
        configs = self.get_configs()
        with self.assertRaises(nidaqmx.errors.DaqError):
            configs["instruments_setup"][
                "activation_instrument"] = "invalid type"
            NationalInstrumentsSetup(configs)

    def test_init_fail_devices_type(self):
        """
        AssertionError is raised if the device names are of an invalid type
        """
        configs = self.get_configs()
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

    @unittest.skipUnless(brainspy.TEST_MODE == "HARDWARE_CDAQ"
                         or brainspy.TEST_MODE == "HARDWARE_NIDAQ",
                         "Hardware test is skipped for simulation setup.")
    def test_init_configs(self):
        """
        Test to initialize the configs of the setup
        """
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

    @unittest.skipUnless(brainspy.TEST_MODE == "HARDWARE_CDAQ"
                         or brainspy.TEST_MODE == "HARDWARE_NIDAQ",
                         "Hardware test is skipped for simulation setup.")
    def test_sampling_configs(self):
        """
        Test to initialize the sampling configs
        """
        try:
            setup = NationalInstrumentsSetup(self.get_configs())
            setup.init_sampling_configs()
        except (Exception):
            self.fail("Could not initialize the sampling configs")
        else:
            self.assertIsNotNone(setup.io_point_difference)

    @unittest.skipUnless(brainspy.TEST_MODE == "HARDWARE_CDAQ"
                         or brainspy.TEST_MODE == "HARDWARE_NIDAQ",
                         "Hardware test is skipped for simulation setup.")
    def test_sampling_configs_fail(self):
        """
        AssertionError is raised if the activation_sampling_frequency % readout_sampling frequency !=0
        """
        configs = self.get_configs()
        configs["instruments_setup"]["activation_sampling_frequency"] = 3000
        configs["instruments_setup"]["readout_sampling_frequency"] = 2000
        with self.assertRaises(AssertionError):
            setup = NationalInstrumentsSetup(self.get_configs)
            setup.init_sampling_configs()

    @unittest.skipUnless(brainspy.TEST_MODE == "HARDWARE_CDAQ"
                         or brainspy.TEST_MODE == "HARDWARE_NIDAQ",
                         "Hardware test is skipped for simulation setup.")
    def test_init_tasks(self):
        """
        Test to initialize the tasks
        """
        try:
            setup = NationalInstrumentsSetup(self.get_configs())
            setup.init_tasks()
        except (Exception):
            self.fail("Could not initialize the tasks")
        else:
            self.assertIsInstance(setup.tasks_driver, IOTasksManager)

    @unittest.skipUnless(brainspy.TEST_MODE == "HARDWARE_CDAQ"
                         or brainspy.TEST_MODE == "HARDWARE_NIDAQ",
                         "Hardware test is skipped for simulation setup.")
    def test_init_semaphore(self):
        """
        Test to initialize the semaphore
        """
        try:
            setup = NationalInstrumentsSetup(self.get_configs())
            setup.init_semaphore()
        except (Exception):
            self.fail("Could not initialize the semaphore for the tasks")

    @unittest.skipUnless(brainspy.TEST_MODE == "HARDWARE_CDAQ"
                         or brainspy.TEST_MODE == "HARDWARE_NIDAQ",
                         "Hardware test is skipped for simulation setup.")
    def test_enable_os_signals(self):
        """
        Test to enable the OS signals
        """
        try:
            setup = NationalInstrumentsSetup(self.get_configs())
            setup.enable_os_signals()
        except (Exception):
            self.fail("Could not enable_os_signals")

    @unittest.skipUnless(brainspy.TEST_MODE == "HARDWARE_CDAQ"
                         or brainspy.TEST_MODE == "HARDWARE_NIDAQ",
                         "Hardware test is skipped for simulation setup.")
    def test_disable_os_signals(self):
        """
        Test to disable the OS signals
        """
        try:
            setup = NationalInstrumentsSetup(self.get_configs())
            setup.disable_os_signals()
        except (Exception):
            self.fail("Could not disable_os_signals")

    @unittest.skipUnless(brainspy.TEST_MODE == "HARDWARE_CDAQ"
                         or brainspy.TEST_MODE == "HARDWARE_NIDAQ",
                         "Hardware test is skipped for simulation setup.")
    def test_os_signal_handler(self):
        """
        Test for the OS signal Handler
        """
        try:
            setup = NationalInstrumentsSetup(self.get_configs())
            setup.os_signal_handler(signum=100)
        except (Exception):
            self.fail("Could not initialize the os_signal_handler")

    @unittest.skipUnless(brainspy.TEST_MODE == "HARDWARE_CDAQ"
                         or brainspy.TEST_MODE == "HARDWARE_NIDAQ",
                         "Hardware test is skipped for simulation setup.")
    def test_get_amplification_value(self):
        """
        Test to get the amplification value
        """
        try:
            setup = NationalInstrumentsSetup(self.get_configs())
            val = setup.get_amplification_value()
        except (Exception):
            self.fail("Could not get_amplification_value")
        else:
            self.assertEqual(val, 50.5)

    @unittest.skipUnless(brainspy.TEST_MODE == "HARDWARE_CDAQ"
                         or brainspy.TEST_MODE == "HARDWARE_NIDAQ",
                         "Hardware test is skipped for simulation setup.")
    def test_read_security_checks(self):
        """
        Test to read the security checks
        """
        configs = self.get_configs()
        try:
            setup = NationalInstrumentsSetup()
            y = np.random.rand(
                len(configs["instruments_setup"]["activation_channels"]),
                random.randint(
                    1,
                    len(configs["instruments_setup"]["activation_channels"])))
            setup.read_security_checks(y)
        except (Exception):
            self.fail("Could not read_security_checks")

    @unittest.skipUnless(brainspy.TEST_MODE == "HARDWARE_CDAQ"
                         or brainspy.TEST_MODE == "HARDWARE_NIDAQ",
                         "Hardware test is skipped for simulation setup.")
    def test_close_tasks(self):
        """
        Test to close all tasks on the setup
        """
        try:
            setup = NationalInstrumentsSetup(self.get_configs())
            setup.close_tasks(signum=100)
        except (Exception):
            self.fail("Could not close_tasks")

    @unittest.skipUnless(brainspy.TEST_MODE == "HARDWARE_CDAQ"
                         or brainspy.TEST_MODE == "HARDWARE_NIDAQ",
                         "Hardware test is skipped for simulation setup.")
    def test__read_data(self):
        """
        Test to read data from the device
        """
        configs = self.get_configs()
        try:
            setup = NationalInstrumentsSetup(self.get_configs())
            y = np.random.rand(
                len(configs["instruments_setup"]["activation_channels"]),
                random.randint(
                    1,
                    len(configs["instruments_setup"]["activation_channels"])))
            val = setup._read_data(y)
        except (Exception):
            self.fail("Could not _read_data from device")
        else:
            self.assertIsNotNone(val)

    @unittest.skipUnless(brainspy.TEST_MODE == "HARDWARE_CDAQ"
                         or brainspy.TEST_MODE == "HARDWARE_NIDAQ",
                         "Hardware test is skipped for simulation setup.")
    def test_is_hardware(self):
        """
        Test to check if the device is connected to hardware
        """
        try:
            setup = NationalInstrumentsSetup(self.get_configs())
            val = setup.is_hardware()
        except (Exception):
            self.fail("Could not check if is_hardware")
        else:
            self.assertTrue(type(val) == bool)

    @unittest.skipUnless(brainspy.TEST_MODE == "HARDWARE_CDAQ"
                         or brainspy.TEST_MODE == "HARDWARE_NIDAQ",
                         "Hardware test is skipped for simulation setup.")
    def test_set_timeout(self):
        """
        Tests to set a timeout for the device
        """
        try:
            setup = NationalInstrumentsSetup(self.get_configs())
            setup.set_timeout()
        except (Exception):
            self.fail("Could not set timeout for device")
        else:
            timeout = setup.offsetted_points_to_write * setup.io_point_difference
            test_timeout = (math.ceil(timeout) + 10)
            self.assertEqual(setup.timeout, test_timeout)

        try:
            setup = NationalInstrumentsSetup(self.get_configs())
            setup.set_timeout(100)
        except (Exception):
            self.fail("Could not set timeout for device")
        else:
            self.assertEqual(setup.timeout, 100)

        try:
            setup = NationalInstrumentsSetup(self.get_configs())
            setup.set_timeout(None)
        except (Exception):
            self.fail("Could not set timeout for device")
        else:
            timeout = setup.offsetted_points_to_write * setup.io_point_difference
            test_timeout = (math.ceil(timeout) + 10)
            self.assertEqual(setup.timeout, test_timeout)

    @unittest.skipUnless(brainspy.TEST_MODE == "HARDWARE_CDAQ"
                         or brainspy.TEST_MODE == "HARDWARE_NIDAQ",
                         "Hardware test is skipped for simulation setup.")
    def test_calculate_io_points(self):
        """
        Test to calculate the number of reads/writes to the device
        """
        try:
            setup = NationalInstrumentsSetup(self.get_configs())
            setup.calculate_io_points(1000)
        except (Exception):
            self.fail("Could not calculate_io_points")
        else:
            self.assertIsNotNone(setup.offsetted_points_to_read)
            self.assertIsNotNone(setup.offsetted_points_to_write)

    @unittest.skipUnless(brainspy.TEST_MODE == "HARDWARE_CDAQ"
                         or brainspy.TEST_MODE == "HARDWARE_NIDAQ",
                         "Hardware test is skipped for simulation setup.")
    def test_set_io_configs(self):
        """
        Test to set the IO configurations for the device
        """
        try:
            setup = NationalInstrumentsSetup(self.get_configs())
            setup.set_io_configs(1000, 100)
        except (Exception):
            self.fail("Could not set_io_configs")

    @unittest.skipUnless(brainspy.TEST_MODE == "HARDWARE_CDAQ"
                         or brainspy.TEST_MODE == "HARDWARE_NIDAQ",
                         "Hardware test is skipped for simulation setup.")
    def test_read_data(self):
        """
        Test to read data from the device
        """
        configs = self.get_configs()
        try:
            setup = NationalInstrumentsSetup(self.get_configs())
            y = np.random.rand(
                len(configs["instruments_setup"]["activation_channels"]),
                random.randint(
                    1,
                    len(configs["instruments_setup"]["activation_channels"])))
            val = setup.read_data(y)
        except (Exception):
            self.fail("Could not read_data from device")
        else:
            self.assertIsNotNone(val)

    @unittest.skipUnless(brainspy.TEST_MODE == "HARDWARE_CDAQ"
                         or brainspy.TEST_MODE == "HARDWARE_NIDAQ",
                         "Hardware test is skipped for simulation setup.")
    def test_average_point_difference(self):
        """
        Test to calculate the averages for all the points that were read per point
        """
        configs = self.get_configs()
        try:
            setup = NationalInstrumentsSetup(self.get_configs())
            y = np.random.rand(
                len(configs["instruments_setup"]["activation_channels"]),
                random.randint(
                    1,
                    len(configs["instruments_setup"]["activation_channels"])))
            val = setup.average_point_difference(y)
        except (Exception):
            self.fail("Could not get average_point_difference")
        else:
            self.assertIsNotNone(val)

    @unittest.skipUnless(brainspy.TEST_MODE == "HARDWARE_CDAQ"
                         or brainspy.TEST_MODE == "HARDWARE_NIDAQ",
                         "Hardware test is skipped for simulation setup.")
    def test_process_output_data(self):
        """
        Test to process the output data
        """
        configs = self.get_configs()
        try:
            setup = NationalInstrumentsSetup(self.get_configs())
            data = np.random.rand(
                len(configs["instruments_setup"]["activation_channels"]),
                random.randint(
                    1,
                    len(configs["instruments_setup"]["activation_channels"])))
            val = setup.process_output_data(data)
        except (Exception):
            self.fail("Could not get average_point_difference")
        else:
            self.assertIsNotNone(val)


if __name__ == "__main__":

    testobj = Setup_Test()
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
