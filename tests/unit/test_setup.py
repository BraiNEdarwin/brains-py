import unittest
import numpy as np
import nidaqmx
import random
import brainspy
from brainspy.processors.hardware.drivers.ni.tasks import IOTasksManager
from brainspy.processors.hardware.drivers.ni.setup import NationalInstrumentsSetup
from tests.unit.testing_utils import get_configs


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
    @unittest.skipUnless(brainspy.__TEST_MODE__ == "HARDWARE_CDAQ"
                         or brainspy.__TEST_MODE__ == "HARDWARE_NIDAQ",
                         "Hardware test is skipped for simulation setup.")
    def test_close_tasks(self):
        """
        Test to close all tasks on the setup
        """
        try:
            setup = NationalInstrumentsSetup(get_configs())
            setup.close_tasks()
        except (Exception):
            self.fail("Could not close_tasks")

    @unittest.skipUnless(brainspy.__TEST_MODE__ == "HARDWARE_CDAQ"
                         or brainspy.__TEST_MODE__ == "HARDWARE_NIDAQ",
                         "Hardware test is skipped for simulation setup.")
    def test_init(self):
        """
        Test to check correct initialization of the Setup
        """
        setup = None
        try:
            configs = get_configs()
            setup = NationalInstrumentsSetup(configs)
        except (Exception):
            if setup is not None:
                setup.close_tasks()
            self.fail("Could not initialize the NationalInstrumentsSetup")
        else:
            self.assertEqual(setup.inversion, -1)
            self.assertEqual(setup.last_points_to_write_val, -1)
            self.assertIsNone(setup.data_results)
            self.assertIsNone(setup.offsetted_points_to_write)
            self.assertIsNone(setup.timeout)
            self.assertIsInstance(setup.tasks_driver, IOTasksManager)
        if setup is not None:
            setup.close_tasks()

    @unittest.skipUnless(brainspy.__TEST_MODE__ == "HARDWARE_CDAQ"
                         or brainspy.__TEST_MODE__ == "HARDWARE_NIDAQ",
                         "Hardware test is skipped for simulation setup.")
    def test_forward_numpy(self):
        """
        Test to check correct initialization of the Setup
        """
        setup = None
        try:
            configs = get_configs()
            setup = NationalInstrumentsSetup(configs)
            setup.forward_numpy()
        except (Exception):
            if setup is not None:
                setup.close_tasks()
            self.fail("Could not initialize the NationalInstrumentsSetup")
        if setup is not None:
            setup.close_tasks()

    @unittest.skipUnless(brainspy.__TEST_MODE__ == "HARDWARE_CDAQ"
                         or brainspy.__TEST_MODE__ == "HARDWARE_NIDAQ",
                         "Hardware test is skipped for simulation setup.")
    def test_signals(self):
        """
        Test to check correct initialization of the Setup
        """
        setup = None
        with self.assertRaises(SystemExit):
            configs = get_configs()
            setup = NationalInstrumentsSetup(configs)
            setup.os_signal_handler(None)
        if setup is not None:
            setup.close_tasks()

    @unittest.skipUnless(brainspy.__TEST_MODE__ == "HARDWARE_CDAQ"
                         or brainspy.__TEST_MODE__ == "HARDWARE_NIDAQ",
                         "Hardware test is skipped for simulation setup.")
    def test_init_sampling_frequency(self):
        """
        Test to check correct initialization of the Setup
        """
        setup = None
        try:
            configs = get_configs()
            configs["instruments_setup"][
                "readout_sampling_frequency"] = configs["instruments_setup"][
                    "activation_sampling_frequency"]  #int(configs["instruments_setup"]["activation_sampling_frequency"] / 2)
            setup = NationalInstrumentsSetup(configs)
        except (Exception):
            if setup is not None:
                setup.close_tasks()
            self.fail("Could not initialize the NationalInstrumentsSetup")
        else:
            self.assertEqual(setup.inversion, -1)
            self.assertEqual(setup.last_points_to_write_val, -1)
            self.assertIsNone(setup.data_results)
            self.assertIsNone(setup.offsetted_points_to_write)
            self.assertIsNone(setup.timeout)
            self.assertIsInstance(setup.tasks_driver, IOTasksManager)
        if setup is not None:
            setup.close_tasks()

    @unittest.skipUnless(brainspy.__TEST_MODE__ == "HARDWARE_CDAQ"
                         or brainspy.__TEST_MODE__ == "HARDWARE_NIDAQ",
                         "Hardware test is skipped for simulation setup.")
    def test_init_multiple(self):
        """
        Test to check correct initialization of the Setup
        """
        setup = None
        try:
            configs = get_configs()
            # TODO Specify the Activation channels (pin numbers)
            # For example, [1,2,3,4,5,6,7]
            configs['inverted_output'] = False
            configs["instruments_setup"]["multiple_devices"] = True
            configs["instruments_setup"]["A"] = {}

            configs['instruments_setup']["A"][
                "activation_instrument"] = configs['instruments_setup'][
                    "activation_instrument"]
            del configs['instruments_setup']['activation_instrument']

            configs["instruments_setup"]["A"]["activation_channels"] = configs[
                'instruments_setup']['activation_channels']
            del configs['instruments_setup']['activation_channels']

            configs["instruments_setup"]["A"][
                "activation_channel_mask"] = configs["instruments_setup"][
                    "activation_channel_mask"]
            del configs["instruments_setup"]["activation_channel_mask"]

            # TODO Specify the activation Voltage ranges
            # For example, [[-1.2, 0.6],[-1.2, 0.6],[-1.2, 0.6],[-1.2, 0.6],[-1.2, 0.6],[-0.7, 0.3],[-0.7, 0.3]]
            configs["instruments_setup"]["A"][
                "activation_voltage_ranges"] = configs["instruments_setup"][
                    "activation_voltage_ranges"]
            del configs["instruments_setup"]["activation_voltage_ranges"]

            # TODO Specify the name of the Readout Instrument
            configs["instruments_setup"]["A"]["readout_instrument"] = configs[
                "instruments_setup"]["readout_instrument"]
            del configs["instruments_setup"]["readout_instrument"]
            # TODO Specify the readout channels
            # For example, [4]
            configs["instruments_setup"]["A"]["readout_channels"] = configs[
                "instruments_setup"]["readout_channels"]
            del configs["instruments_setup"]["readout_channels"]

            setup = NationalInstrumentsSetup(configs)
        except (Exception):
            if setup is not None:
                setup.close_tasks()
            self.fail("Could not initialize the NationalInstrumentsSetup")
        else:
            self.assertEqual(setup.inversion, 1)
            if setup is not None:
                setup.close_tasks()

    @unittest.skipUnless(brainspy.__TEST_MODE__ == "HARDWARE_CDAQ"
                         or brainspy.__TEST_MODE__ == "HARDWARE_NIDAQ",
                         "Hardware test is skipped for simulation setup.")
    def test_init_fail_amplification(self):
        """
        AssertionError is raised if the amplification value is of an invalid type
        """
        configs = get_configs()
        with self.assertRaises(AssertionError):
            configs["amplification"] = "invalid type"
            NationalInstrumentsSetup(configs)
        with self.assertRaises(AssertionError):
            configs["amplification"] = {}
            NationalInstrumentsSetup(configs)
        with self.assertRaises(AssertionError):
            configs["amplification"] = None
            NationalInstrumentsSetup(configs)
        with self.assertRaises(AssertionError):
            del configs["amplification"]
            NationalInstrumentsSetup(configs)

    @unittest.skipUnless(brainspy.__TEST_MODE__ == "HARDWARE_CDAQ"
                         or brainspy.__TEST_MODE__ == "HARDWARE_NIDAQ",
                         "Hardware test is skipped for simulation setup.")
    def test_init_fail_devices(self):
        """
        nidaqmx.errors.DaqError is raised if the device names is incorrect
        """
        configs = get_configs()
        setup = None
        with self.assertRaises(nidaqmx.errors.DaqError):
            configs["instruments_setup"][
                "activation_instrument"] = "invalid type"
            setup = NationalInstrumentsSetup(configs)
        if setup is not None:
            setup.close_tasks()

        with self.assertRaises(AssertionError):
            del configs["instruments_setup"]["activation_instrument"]
            setup = NationalInstrumentsSetup(configs)
        if setup is not None:
            setup.close_tasks()

    @unittest.skipUnless(brainspy.__TEST_MODE__ == "HARDWARE_CDAQ"
                         or brainspy.__TEST_MODE__ == "HARDWARE_NIDAQ",
                         "Hardware test is skipped for simulation setup.")
    def test_init_fail_devices_type(self):
        """
        AssertionError is raised if the device names are of an invalid type
        """
        configs = get_configs()

        setup = None
        with self.assertRaises(AssertionError):
            configs["instruments_setup"]["readout_instrument"] = -5
            setup = NationalInstrumentsSetup(configs)
        if setup is not None:
            setup.close_tasks()

        setup = None
        with self.assertRaises(AssertionError):
            del configs["instruments_setup"]["readout_instrument"]
            NationalInstrumentsSetup(configs)
        if setup is not None:
            setup.close_tasks()

        configs = get_configs()
        setup = None
        with self.assertRaises(AssertionError):
            configs["instruments_setup"]["trigger_source"] = 100
            setup = NationalInstrumentsSetup(configs)
        if setup is not None:
            setup.close_tasks()

        configs = get_configs()
        setup = None
        with self.assertRaises(AssertionError):
            del configs["instruments_setup"]["trigger_source"]
            setup = NationalInstrumentsSetup(configs)
        if setup is not None:
            setup.close_tasks()

        # configs = get_configs()
        # setup = None
        # with self.assertRaises(AssertionError):
        #     del configs["instruments_setup"]["trigger_source"]
        #     setup = NationalInstrumentsSetup(configs)
        # if setup is not None:
        #     setup.close_tasks()

        configs = get_configs()
        setup = None
        with self.assertRaises(AssertionError):
            configs["instruments_setup"]["activation_sampling_frequency"] = {}
            setup = NationalInstrumentsSetup(configs)
        if setup is not None:
            setup.close_tasks()

        configs = get_configs()
        setup = None
        with self.assertRaises(AssertionError):
            del configs["instruments_setup"]["activation_sampling_frequency"]
            setup = NationalInstrumentsSetup(configs)
        if setup is not None:
            setup.close_tasks()

        configs = get_configs()
        setup = None
        with self.assertRaises(AssertionError):
            configs["instruments_setup"]["readout_sampling_frequency"] = [
                1, 2, 3, 4
            ]
            setup = NationalInstrumentsSetup(configs)
        if setup is not None:
            setup.close_tasks()

        configs = get_configs()
        setup = None
        with self.assertRaises(AssertionError):
            del configs["instruments_setup"]["readout_sampling_frequency"]
            setup = NationalInstrumentsSetup(configs)
        if setup is not None:
            setup.close_tasks()

        configs = get_configs()
        setup = None
        with self.assertRaises(AssertionError):
            configs["instruments_setup"]["readout_channels"] = "invalidtype"
            setup = NationalInstrumentsSetup(configs)
        if setup is not None:
            setup.close_tasks()

        configs = get_configs()
        setup = None
        with self.assertRaises(AssertionError):
            del configs["instruments_setup"]["readout_channels"]
            setup = NationalInstrumentsSetup(configs)
        if setup is not None:
            setup.close_tasks()

    @unittest.skipUnless(brainspy.__TEST_MODE__ == "HARDWARE_CDAQ"
                         or brainspy.__TEST_MODE__ == "HARDWARE_NIDAQ",
                         "Hardware test is skipped for simulation setup.")
    def test_init_activation_channels_error_1(self):
        """
        AssertionError is raised if the channel_names are not of type - list or numpy array
        """
        configs = get_configs()
        setup = None
        with self.assertRaises(AssertionError):
            configs["instruments_setup"]["activation_channels"] = {}
            setup = NationalInstrumentsSetup(configs)
        if setup is not None:
            setup.close_tasks()

        configs = get_configs()
        setup = None
        with self.assertRaises(AssertionError):
            configs["instruments_setup"]["activation_channels"] = 100
            NationalInstrumentsSetup(configs)
        if setup is not None:
            setup.close_tasks()

        configs = get_configs()
        setup = None
        with self.assertRaises(AssertionError):
            configs["instruments_setup"][
                "activation_channels"] = "invalid type"
            setup = NationalInstrumentsSetup(configs)
        if setup is not None:
            setup.close_tasks()

        configs = get_configs()
        setup = None
        with self.assertRaises(AssertionError):
            configs["instruments_setup"][
                "activation_channels"] = "invalid type"
            setup = NationalInstrumentsSetup(configs)
        if setup is not None:
            setup.close_tasks()

    @unittest.skipUnless(brainspy.__TEST_MODE__ == "HARDWARE_CDAQ"
                         or brainspy.__TEST_MODE__ == "HARDWARE_NIDAQ",
                         "Hardware test is skipped for simulation setup.")
    def test_init_activation_voltages(self):
        """
        AssertionError is raised if the voltage_ranges are not of type - list or numpy array
        """
        configs = get_configs()
        setup = None
        with self.assertRaises(AssertionError):
            configs["instruments_setup"]["activation_voltage_ranges"] = {}
            setup = NationalInstrumentsSetup(configs)
        if setup is not None:
            setup.close_tasks()

        configs = get_configs()
        setup = None
        with self.assertRaises(AssertionError):
            configs["instruments_setup"]["activation_voltage_ranges"] = 100
            setup = NationalInstrumentsSetup(configs)
        if setup is not None:
            setup.close_tasks()

        configs = get_configs()
        setup = None
        with self.assertRaises(AssertionError):
            configs["instruments_setup"][
                "activation_voltage_ranges"] = "invalid type"
            setup = NationalInstrumentsSetup(configs)
        if setup is not None:
            setup.close_tasks()

        configs = get_configs()
        setup = None
        with self.assertRaises(AssertionError):
            del configs["instruments_setup"]["activation_voltage_ranges"]
            setup = NationalInstrumentsSetup(configs)
        if setup is not None:
            setup.close_tasks()

    @unittest.skipUnless(brainspy.__TEST_MODE__ == "HARDWARE_CDAQ"
                         or brainspy.__TEST_MODE__ == "HARDWARE_NIDAQ",
                         "Hardware test is skipped for simulation setup.")
    def test_init_activation_channels_error_different_size(self):
        """
        AssertionError is raised if the length of channel names is not equal to the
        length of voltage_ranges
        """
        configs = get_configs()
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
        setup = None
        with self.assertRaises(AssertionError):
            setup = NationalInstrumentsSetup(configs)
        if setup is not None:
            setup.close_tasks()

    @unittest.skipUnless(brainspy.__TEST_MODE__ == "HARDWARE_CDAQ"
                         or brainspy.__TEST_MODE__ == "HARDWARE_NIDAQ",
                         "Hardware test is skipped for simulation setup.")
    def test_init_activation_channels_error_4(self):
        """
        AssertionError is raised if each voltage range is not a list or a numpy array
        """
        configs = get_configs()
        configs["instruments_setup"]["activation_voltage_ranges"] = [
            [-1.2, 0.6],
            [-1.2, 0.6],
            [-1.2, 0.6],
            [-1.2, 0.6],
            "Invalid type",
            [-0.7, 0.3],
            [-0.7, 0.3],
        ]
        setup = None
        with self.assertRaises(AssertionError):
            setup = NationalInstrumentsSetup(configs)
        if setup is not None:
            setup.close_tasks()

    @unittest.skipUnless(brainspy.__TEST_MODE__ == "HARDWARE_CDAQ"
                         or brainspy.__TEST_MODE__ == "HARDWARE_NIDAQ",
                         "Hardware test is skipped for simulation setup.")
    def test_init_activation_channels_error_5(self):
        """
        AssertionError is raised if each voltage range is not a list or a numpy array of length = 2
        """
        configs = get_configs()
        configs["instruments_setup"]["activation_voltage_ranges"] = [
            [-1.2, 0.6],
            [-1.2, 0.6],
            [-1.2, 0.6],
            [-1.2, 0.6],
            [-1.2, 0.6, 1.2],
            [-0.7, 0.3],
            [-0.7, 0.3],
        ]
        setup = None
        with self.assertRaises(AssertionError):
            setup = NationalInstrumentsSetup(configs)
        if setup is not None:
            setup.close_tasks()

    @unittest.skipUnless(brainspy.__TEST_MODE__ == "HARDWARE_CDAQ"
                         or brainspy.__TEST_MODE__ == "HARDWARE_NIDAQ",
                         "Hardware test is skipped for simulation setup.")
    def test_init_activation_channels_error_6(self):
        """
        AssertionError is raised if a voltage range does not contain an int/float type value
        """
        configs = get_configs()
        configs["instruments_setup"]["activation_voltage_ranges"] = [
            [-1.2, 0.6],
            [-1.2, 0.6],
            [-1.2, 0.6],
            [-1.2, 0.6],
            [-1.2, "Invalid type"],
            [-0.7, 0.3],
            [-0.7, 0.3],
        ]
        setup = None
        with self.assertRaises(AssertionError):
            setup = NationalInstrumentsSetup(configs)
        if setup is not None:
            setup.close_tasks()

    @unittest.skipUnless(brainspy.__TEST_MODE__ == "HARDWARE_CDAQ"
                         or brainspy.__TEST_MODE__ == "HARDWARE_NIDAQ",
                         "Hardware test is skipped for simulation setup.")
    def test_init_configs(self):
        """
        Test to initialize the configs of the setup
        """
        setup = None
        try:
            setup = NationalInstrumentsSetup(get_configs())
        except (Exception):
            if setup is not None:
                setup.close_tasks()
            self.fail("Could not initialize the configs")
        else:
            self.assertEqual(setup.inversion, -1)
            self.assertEqual(setup.last_points_to_write_val, -1)
            self.assertIsNone(setup.data_results)
            self.assertIsNone(setup.offsetted_points_to_write)
            self.assertIsNone(setup.timeout)
        if setup is not None:
            setup.close_tasks()

    @unittest.skipUnless(brainspy.__TEST_MODE__ == "HARDWARE_CDAQ"
                         or brainspy.__TEST_MODE__ == "HARDWARE_NIDAQ",
                         "Hardware test is skipped for simulation setup.")
    def test_sampling_configs(self):
        """
        Test to initialize the sampling configs
        """
        setup = None
        try:
            configs = get_configs()
            setup = NationalInstrumentsSetup(configs)
            setup.init_sampling_configs(configs)
        except (Exception):
            if setup is not None:
                setup.close_tasks()
            self.fail("Could not initialize the sampling configs")
        else:
            self.assertIsNotNone(setup.io_point_difference)
        if setup is not None:
            setup.close_tasks()

    @unittest.skipUnless(brainspy.__TEST_MODE__ == "HARDWARE_CDAQ"
                         or brainspy.__TEST_MODE__ == "HARDWARE_NIDAQ",
                         "Hardware test is skipped for simulation setup.")
    def test_sampling_configs_fail(self):
        """
        AssertionError is raised if the activation_sampling_frequency % readout_sampling frequency !=0
        """
        configs = get_configs()
        configs["instruments_setup"]["activation_sampling_frequency"] = 3.8
        configs["instruments_setup"]["readout_sampling_frequency"] = 2.6
        setup = None
        with self.assertRaises(AssertionError):
            setup = NationalInstrumentsSetup(configs)
            setup.init_sampling_configs(configs)
        if setup is not None:
            setup.close_tasks()

    @unittest.skipUnless(brainspy.__TEST_MODE__ == "HARDWARE_CDAQ"
                         or brainspy.__TEST_MODE__ == "HARDWARE_NIDAQ",
                         "Hardware test is skipped for simulation setup.")
    def test_init_tasks(self):
        """
        Test to initialize the tasks
        """
        try:
            configs = get_configs()
            setup = None
            setup = NationalInstrumentsSetup(configs)
            setup.close_tasks()
            setup.init_tasks(configs)
        except (Exception):
            if setup is not None:
                setup.close_tasks()
            self.fail("Could not initialize the tasks")
        else:
            self.assertIsInstance(setup.tasks_driver, IOTasksManager)
        if setup is not None:
            setup.close_tasks()

    @unittest.skipUnless(brainspy.__TEST_MODE__ == "HARDWARE_CDAQ"
                         or brainspy.__TEST_MODE__ == "HARDWARE_NIDAQ",
                         "Hardware test is skipped for simulation setup.")
    def test_init_semaphore(self):
        """
        Test to initialize the semaphore
        """
        try:
            setup = None
            setup = NationalInstrumentsSetup(get_configs())
            setup.init_semaphore()
        except (Exception):
            if setup is not None:
                setup.close_tasks()
            self.fail("Could not initialize the semaphore for the tasks")
        else:
            if setup is not None:
                setup.close_tasks()

    @unittest.skipUnless(brainspy.__TEST_MODE__ == "HARDWARE_CDAQ"
                         or brainspy.__TEST_MODE__ == "HARDWARE_NIDAQ",
                         "Hardware test is skipped for simulation setup.")
    def test_enable_os_signals(self):
        """
        Test to enable the OS signals
        """
        try:
            setup = None
            setup = NationalInstrumentsSetup(get_configs())
            setup.enable_os_signals()
        except (Exception):
            if setup is not None:
                setup.close_tasks()
            self.fail("Could not enable_os_signals")
        else:
            if setup is not None:
                setup.close_tasks()

    @unittest.skipUnless(brainspy.__TEST_MODE__ == "HARDWARE_CDAQ"
                         or brainspy.__TEST_MODE__ == "HARDWARE_NIDAQ",
                         "Hardware test is skipped for simulation setup.")
    def test_disable_os_signals(self):
        """
        Test to disable the OS signals
        """
        try:
            setup = None
            setup = NationalInstrumentsSetup(get_configs())
            setup.disable_os_signals()
        except (Exception):
            if setup is not None:
                setup.close_tasks()
            self.fail("Could not disable_os_signals")
        else:
            if setup is not None:
                setup.close_tasks()

    # @unittest.skipUnless(brainspy.__TEST_MODE__ == "HARDWARE_CDAQ"
    #                      or brainspy.__TEST_MODE__ == "HARDWARE_NIDAQ",
    #                      "Hardware test is skipped for simulation setup.")
    # def test_os_signal_handler(self):
    #     """
    #     Test for the OS signal Handler
    #     """
    #     try:
    #         setup = None
    #         setup = NationalInstrumentsSetup(get_configs())
    #         setup.os_signal_handler(signum=100)
    #     except (Exception):
    #         if setup is not None:
    #             setup.close_tasks()
    #         self.fail("Could not initialize the os_signal_handler")
    #     else:
    #         if setup is not None:
    #             setup.close_tasks()

    @unittest.skipUnless(brainspy.__TEST_MODE__ == "HARDWARE_CDAQ"
                         or brainspy.__TEST_MODE__ == "HARDWARE_NIDAQ",
                         "Hardware test is skipped for simulation setup.")
    def test_get_amplification_value(self):
        """
        Test to get the amplification value
        """
        try:
            setup = None
            setup = NationalInstrumentsSetup(get_configs())
            val = setup.get_amplification_value()
        except (Exception):
            if setup is not None:
                setup.close_tasks()
            self.fail("Could not get_amplification_value")
        else:
            self.assertEqual(val, get_configs()['amplification'])
            if setup is not None:
                setup.close_tasks()

    @unittest.skipUnless(brainspy.__TEST_MODE__ == "HARDWARE_CDAQ"
                         or brainspy.__TEST_MODE__ == "HARDWARE_NIDAQ",
                         "Hardware test is skipped for simulation setup.")
    def test_read_security_checks(self):
        """
        Test to read the security checks
        """
        configs = get_configs()
        try:
            setup = None
            setup = NationalInstrumentsSetup(configs)
            y = np.random.rand(
                len(configs["instruments_setup"]["activation_channels"]),
                random.randint(1, 100))
            y[:, 0] = 0
            y[:, -1] = 0
            setup.read_security_checks(y)
        except (Exception):
            if setup is not None:
                setup.close_tasks()
            self.fail("Could not read_security_checks")
        else:
            if setup is not None:
                setup.close_tasks()

    @unittest.skipUnless(brainspy.__TEST_MODE__ == "HARDWARE_CDAQ"
                         or brainspy.__TEST_MODE__ == "HARDWARE_NIDAQ",
                         "Hardware test is skipped for simulation setup.")
    def test_is_hardware(self):
        """
        Test to check if the device is connected to hardware
        """
        try:
            setup = None
            setup = NationalInstrumentsSetup(get_configs())
            val = setup.is_hardware()
        except (Exception):
            if setup is not None:
                setup.close_tasks()
            self.fail("Could not check if is_hardware")
        else:
            self.assertTrue(type(val) == bool)
            if setup is not None:
                setup.close_tasks()

    # @unittest.skipUnless(brainspy.__TEST_MODE__ == "HARDWARE_CDAQ"
    #                      or brainspy.__TEST_MODE__ == "HARDWARE_NIDAQ",
    #                      "Hardware test is skipped for simulation setup.")
    # def test_set_timeout(self):
    #     """
    #     Tests to set a timeout for the device
    #     """
    #     try:
    #         setup = None
    #         setup = NationalInstrumentsSetup(get_configs())
    #         setup.set_timeout()
    #     except (Exception):
    #         if setup is not None:
    #             setup.close_tasks()
    #         self.fail("Could not set timeout for device")
    #     else:
    #         timeout = setup.offsetted_points_to_write * setup.io_point_difference
    #         test_timeout = (math.ceil(timeout) + 10)
    #         self.assertEqual(setup.timeout, test_timeout)
    #         if setup is not None:
    #             setup.close_tasks()

    #     try:
    #         setup = NationalInstrumentsSetup(get_configs())
    #         setup.set_timeout(100)
    #     except (Exception):
    #         self.fail("Could not set timeout for device")
    #     else:
    #         self.assertEqual(setup.timeout, 100)

    #     try:
    #         setup = NationalInstrumentsSetup(get_configs())
    #         setup.set_timeout(None)
    #     except (Exception):
    #         self.fail("Could not set timeout for device")
    #     else:
    #         timeout = setup.offsetted_points_to_write * setup.io_point_difference
    #         test_timeout = (math.ceil(timeout) + 10)
    #         self.assertEqual(setup.timeout, test_timeout)

    @unittest.skipUnless(brainspy.__TEST_MODE__ == "HARDWARE_CDAQ"
                         or brainspy.__TEST_MODE__ == "HARDWARE_NIDAQ",
                         "Hardware test is skipped for simulation setup.")
    def test_calculate_io_points(self):
        """
        Test to calculate the number of reads/writes to the device
        """
        try:
            setup = None
            setup = NationalInstrumentsSetup(get_configs())
            setup.calculate_io_points(1000)
        except (Exception):
            if setup is not None:
                setup.close_tasks()
            self.fail("Could not calculate_io_points")
        else:
            self.assertIsNotNone(setup.offsetted_points_to_read)
            self.assertIsNotNone(setup.offsetted_points_to_write)
            if setup is not None:
                setup.close_tasks()

    @unittest.skipUnless(brainspy.__TEST_MODE__ == "HARDWARE_CDAQ"
                         or brainspy.__TEST_MODE__ == "HARDWARE_NIDAQ",
                         "Hardware test is skipped for simulation setup.")
    def test_set_io_configs(self):
        """
        Test to set the IO configurations for the device
        """
        try:
            setup = None
            setup = NationalInstrumentsSetup(get_configs())
            setup.set_io_configs(1000, 100)
        except (Exception):
            if setup is not None:
                setup.close_tasks()
            self.fail("Could not set_io_configs")
        else:
            if setup is not None:
                setup.close_tasks()

    @unittest.skipUnless(brainspy.__TEST_MODE__ == "HARDWARE_CDAQ"
                         or brainspy.__TEST_MODE__ == "HARDWARE_NIDAQ",
                         "Hardware test is skipped for simulation setup.")
    def test_read_data(self):
        """
        Test to read data from the device
        """
        configs = get_configs()
        try:
            setup = None
            setup = NationalInstrumentsSetup(configs)
            y = np.concatenate(
                (np.linspace(0, 1, 50), np.linspace(1, 0, 50))) / 10
            y = np.broadcast_to(
                y[:, np.newaxis],
                (100, len(
                    configs["instruments_setup"]["activation_channels"])))
            val = setup.read_data(y.T)
        except (Exception):
            if setup is not None:
                setup.close_tasks()
            self.fail("Could not read_data from device")
        else:
            self.assertIsNotNone(val)
            if setup is not None:
                setup.close_tasks()

    @unittest.skipUnless(brainspy.__TEST_MODE__ == "HARDWARE_CDAQ"
                         or brainspy.__TEST_MODE__ == "HARDWARE_NIDAQ",
                         "Hardware test is skipped for simulation setup.")
    def test_average_point_difference(self):
        """
        Test to calculate the averages for all the points that were read per point
        """
        configs = get_configs()
        try:
            setup = None
            setup = NationalInstrumentsSetup(get_configs())
            y = np.random.rand(
                len(configs["instruments_setup"]["activation_channels"]), 100)
            val = setup.average_point_difference(y)
        except (Exception):
            if setup is not None:
                setup.close_tasks()
            self.fail("Could not get average_point_difference")
        else:
            self.assertIsNotNone(val)
            if setup is not None:
                setup.close_tasks()

    @unittest.skipUnless(brainspy.__TEST_MODE__ == "HARDWARE_CDAQ"
                         or brainspy.__TEST_MODE__ == "HARDWARE_NIDAQ",
                         "Hardware test is skipped for simulation setup.")
    def test_process_output_data(self):
        """
        Test to process the output data
        """
        configs = get_configs()
        try:
            setup = None
            setup = NationalInstrumentsSetup(get_configs())
            data = np.random.rand(
                len(configs["instruments_setup"]["activation_channels"]),
                random.randint(
                    1,
                    len(configs["instruments_setup"]["activation_channels"])))
            val = setup.process_output_data(data)
        except (Exception):
            if setup is not None:
                setup.close_tasks()
            self.fail("Could not get average_point_difference")
        else:
            self.assertIsNotNone(val)
            if setup is not None:
                setup.close_tasks()


if __name__ == "__main__":
    unittest.main()