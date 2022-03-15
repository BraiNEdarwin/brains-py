import random
import unittest
import numpy as np
import nidaqmx
import nidaqmx.constants as constants
import nidaqmx.system.device as device

from brainspy.processors.hardware.drivers.ni.tasks import IOTasksManager


class Tasks_Test(unittest.TestCase):
    """
    Tests for tasks with custom configs and no real time rack.
    """

    def get_configs(self):
        """
        Generate the sample configs for the Task Manager
        """
        configs = {}
        configs["processor_type"] = "cdaq_to_nidaq"
        configs["driver"] = {}
        configs["driver"]["sampling_frequency"] = 1000
        configs["driver"]["output_clipping_range"] = [-1, 1]
        configs["driver"]["amplification"] = 50.5
        configs["instruments_setup"] = {}
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
        configs["plateau_length"] = 10
        configs["slope_length"] = 30
        return configs

    def test_init_configs(self):
        """
        Test to initialize the IOTasksManager with some sample configs
        """
        try:
            tasks = IOTasksManager(self.get_configs())
        except (Exception):
            self.fail("Could not initialize the Tasks Manager")
        else:
            self.assertEqual(tasks.local.acquisition_type,
                             constants.AcquisitionType.FINITE)
            self.assertEqual(tasks.local.activation_task, None)
            self.assertEqual(tasks.local.readout_task, None)

    def test_init_configs_keyerror(self):
        """
        Missing keys in configs raises a KeyError
        """
        configs = self.get_configs()
        del configs["instruments_setup"]["activation_channels"]
        with self.assertRaises(KeyError):
            IOTasksManager(configs)

        configs = self.get_configs()
        del configs["instruments_setup"]["activation_voltage_ranges"]
        with self.assertRaises(KeyError):
            IOTasksManager(configs)

        configs = self.get_configs()
        del configs["instruments_setup"]["readout_channels"]
        with self.assertRaises(KeyError):
            IOTasksManager(configs)

    def test_init_invalidtype(self):
        """
        configs should be of type - dict, otherwise an AssertionError is raised
        """
        configs = [1, 2, 3, 4]
        with self.assertRaises(AssertionError):
            IOTasksManager(configs)

        configs = 100
        with self.assertRaises(AssertionError):
            IOTasksManager(configs)

        configs = "self.get_configs()"
        with self.assertRaises(AssertionError):
            IOTasksManager(configs)

    def test_init_tasks(self):
        """
        Test for the init_tasks method of the TaskManager
        """
        try:
            tasks = IOTasksManager(self.get_configs())
            tasks.init_tasks(self.get_configs())
        except (Exception):
            self.fail("Could not initialize the Tasks Manager")
        else:
            self.assertIsNotNone(tasks.activation_channel_names)
            self.assertIsNotNone(tasks.readout_channel_names)
            self.assertEquals(len(tasks.voltage_ranges), 7)
            self.assertIsNotNone(tasks.devices)
            for d in tasks.devices:
                assert isinstance(d, device.Device)

    def test_init_tasks_invalid_input(self):
        """
        Invalid input for the init_tasks method raises an AssertionError
        """
        configs = [1, 2, 3, 4]
        with self.assertRaises(AssertionError):
            tasks = IOTasksManager(self.get_configs())
            tasks.init_tasks(configs)

        configs = 100
        with self.assertRaises(AssertionError):
            tasks = IOTasksManager(self.get_configs())
            tasks.init_tasks(configs)

        configs = "self.get_configs()"
        with self.assertRaises(AssertionError):
            tasks = IOTasksManager(self.get_configs())
            tasks.init_tasks(configs)

    def test_init_activation_channels(self):  #assertions here can be improved
        """
        Test to initialize the activation channels
        """
        try:
            tasks = IOTasksManager(self.get_configs())
            tasks.init_activation_channels(tasks.activation_channel_names,
                                           tasks.voltage_ranges)
        except (Exception):
            self.fail("Could not initialize the activation channels")
        else:
            self.assertIsNotNone(tasks.activation_task.ao_channels)

    def test_init_activation_channels_error_1(self):
        """
        AssertionError is raised if the channel_names are not of type - list or numpy array
        """
        with self.assertRaises(AssertionError):
            tasks = IOTasksManager(self.get_configs())
            tasks.init_activation_channels("Invalid type",
                                           tasks.voltage_ranges)
        with self.assertRaises(AssertionError):
            tasks = IOTasksManager(self.get_configs())
            tasks.init_activation_channels({"Invalid type": "invalid"},
                                           tasks.voltage_ranges)
        with self.assertRaises(AssertionError):
            tasks = IOTasksManager(self.get_configs())
            tasks.init_activation_channels(100, tasks.voltage_ranges)

    def test_init_activation_channels_error_2(self):
        """
        AssertionError is raised if the voltage_ranges are not of type - list or numpy array
        """
        with self.assertRaises(AssertionError):
            tasks = IOTasksManager(self.get_configs())
            tasks.init_activation_channels(tasks.activation_channel_names,
                                           "Invalid type")
        with self.assertRaises(AssertionError):
            tasks = IOTasksManager(self.get_configs())
            tasks.init_activation_channels(tasks.activation_channel_names, 100)
        with self.assertRaises(AssertionError):
            tasks = IOTasksManager(self.get_configs())
            tasks.init_activation_channels(tasks.activation_channel_names, {})

    def test_init_activation_channels_error_3(self):
        """
        AssertionError is raised if the length of channel names is not equal to the
        length of voltage_ranges
        """
        channel_names = [1, 2, 3, 4]
        voltage_ranges = [
            [-1.2, 0.6],
            [-1.2, 0.6],
            [-1.2, 0.6],
            [-1.2, 0.6],
            [-1.2, 0.6],
            [-0.7, 0.3],
            [-0.7, 0.3],
        ]
        with self.assertRaises(AssertionError):
            tasks = IOTasksManager(self.get_configs())
            tasks.init_activation_channels(channel_names, voltage_ranges)

    def test_init_activation_channels_error_4(self):
        """
        AssertionError is raised if each voltage range is not a list or a numpy array
        """
        channel_names = [1, 2, 3, 4, 5, 6, 7]
        voltage_ranges = [
            [-1.2, 0.6],
            [-1.2, 0.6],
            [-1.2, 0.6],
            [-1.2, 0.6],
            "Invalid type",
            [-0.7, 0.3],
            [-0.7, 0.3],
        ]
        with self.assertRaises(AssertionError):
            tasks = IOTasksManager(self.get_configs())
            tasks.init_activation_channels(channel_names, voltage_ranges)

    def test_init_activation_channels_error_5(self):
        """
        AssertionError is raised if each voltage range is not a list or a numpy array of length = 2
        """
        channel_names = [1, 2, 3, 4, 5, 6, 7]
        voltage_ranges = [
            [-1.2, 0.6],
            [-1.2, 0.6],
            [-1.2, 0.6],
            [-1.2, 0.6],
            [-1.2, 0.6, 1.2],
            [-0.7, 0.3],
            [-0.7, 0.3],
        ]
        with self.assertRaises(AssertionError):
            tasks = IOTasksManager(self.get_configs())
            tasks.init_activation_channels(channel_names, voltage_ranges)

    def test_init_activation_channels_error_6(self):
        """
        AssertionError is raised if a voltage range does not contain an int/float type value
        """
        channel_names = [1, 2, 3, 4, 5, 6, 7]
        voltage_ranges = [
            [-1.2, 0.6],
            [-1.2, 0.6],
            [-1.2, 0.6],
            [-1.2, 0.6],
            [-1.2, "Invalid type"],
            [-0.7, 0.3],
            [-0.7, 0.3],
        ]
        with self.assertRaises(AssertionError):
            tasks = IOTasksManager(self.get_configs())
            tasks.init_activation_channels(channel_names, voltage_ranges)

        voltage_ranges = [
            [-1.2, 0.6],
            [-1.2, 0.6],
            ["Invalid type", 0.6],
            [-1.2, 0.6],
            [-1.2, 0.6],
            [-0.7, 0.3],
            [-0.7, 0.3],
        ]
        with self.assertRaises(AssertionError):
            tasks = IOTasksManager(self.get_configs())
            tasks.init_activation_channels(channel_names, voltage_ranges)

    def test_init_readout_channels(self):  #assertions here can be improved
        """
        Test to initialize the readout channels
        """
        try:
            tasks = IOTasksManager(self.get_configs())
            tasks.init_readout_channels(tasks.readout_channel_names)
        except (Exception):
            self.fail("Could not initialize the readout channels")
        else:
            self.assertIsNotNone(tasks.readout_task.ai_channels)

    def test_init_readout_channels_error_1(self):
        """
        AssertionError is raised if an invalid type is provided for the readout channels
        """
        with self.assertRaises(AssertionError):
            tasks = IOTasksManager(self.get_configs())
            tasks.init_readout_channels({})
        with self.assertRaises(AssertionError):
            tasks = IOTasksManager(self.get_configs())
            tasks.init_readout_channels(100)
        with self.assertRaises(AssertionError):
            tasks = IOTasksManager(self.get_configs())
            tasks.init_readout_channels("Invalid type")

    def test_init_readout_channels_error_2(self):
        """
        AssertionError is raised if a String type is not provided for each readout channel name
        """
        with self.assertRaises(AssertionError):
            tasks = IOTasksManager(self.get_configs())
            tasks.init_readout_channels([1, 2, 3, 4, 5])
        with self.assertRaises(AssertionError):
            tasks = IOTasksManager(self.get_configs())
            tasks.init_readout_channels(["1", "2", 3])

    def test_set_sampling_frequencies(self):
        """
        Test to set the sampling frequency with some random values
        """
        try:
            tasks = IOTasksManager(self.get_configs())
            tasks.set_sampling_frequencies(random.randint(1, 1000),
                                           random.randint(1, 1000),
                                           random.randint(1, 1000),
                                           random.randint(1, 1000))
        except (Exception):
            self.fail("Could not set the sampling frequency for these values")

    def test_set_sampling_frequencies_fail(self):
        """
        AssertionError is raised if an integer value is not provided for any of the
        parameters of the set_sampling_frequencies function
        """
        with self.assertRaises(AssertionError):
            tasks = IOTasksManager(self.get_configs())
            tasks.set_sampling_frequencies("Invalid type",
                                           random.randint(1, 1000),
                                           random.randint(1, 1000),
                                           random.randint(1, 1000))
        with self.assertRaises(AssertionError):
            tasks = IOTasksManager(self.get_configs())
            tasks.set_sampling_frequencies(random.randint(1, 1000),
                                           random.randint(1, 1000), {},
                                           random.randint(1, 1000))
        with self.assertRaises(AssertionError):
            tasks = IOTasksManager(self.get_configs())
            tasks.set_sampling_frequencies(random.randint(1, 1000), 55.55,
                                           random.randint(1, 1000),
                                           random.randint(1, 1000))
        with self.assertRaises(AssertionError):
            tasks = IOTasksManager(self.get_configs())
            tasks.set_sampling_frequencies(random.randint(1, 1000),
                                           random.randint(1, 1000),
                                           random.randint(1, 1000),
                                           [1, 3, 2, 4])

    def test_add_synchronisation_channels(self):
        """
        Test to add a synchronisation channel
        """
        try:
            tasks = IOTasksManager(self.get_configs())
            tasks.add_synchronisation_channels(
                "cDAQ1Mod3", "cDAQ1Mod4"
            )  #device name maybe different, also addd another test when device name is wrong
        except (Exception):
            self.fail("Could not add a synchronization channel")
        else:
            self.assertIsNotNone(tasks.activation_task.ao_channels)
            self.assertIsNotNone(tasks.activation_task.ai_channels)

    def test_add_synchronisation_channels_invalid_type(self):
        """
        AssertionError is raised if invalid type for instrument names are provided
        """
        with self.assertRaises(AssertionError):
            tasks = IOTasksManager(self.get_configs())
            tasks.add_synchronisation_channels(100, "cDAQ1Mod4")
        with self.assertRaises(AssertionError):
            tasks = IOTasksManager(self.get_configs())
            tasks.add_synchronisation_channels("cDAQ1Mod3", {})

    def test_read(self):
        """
        Test to read from the device without any params
        """
        try:
            tasks = IOTasksManager(self.get_configs())
            read_data = tasks.read()
        except (Exception):
            self.fail("Could not read any data")
        else:
            self.assertIsNotNone(read_data)

    def test_read_random(self):
        """
        Test to read from the device with set parameters
        """
        try:
            tasks = IOTasksManager(self.get_configs())
            read_data = tasks.read(random.randint(1, 100),
                                   random.uniform(1.0, 100.0))
        except (Exception):
            self.fail("Could not read any data")
        else:
            self.assertIsNotNone(read_data)

    def test_read_fail(self):
        """
        AssertionError is raised if invalid type is provided
        """
        with self.assertRaises(AssertionError):
            tasks = IOTasksManager(self.get_configs())
            tasks.read("Invalid type", random.uniform(1.0, 100.0))

        with self.assertRaises(AssertionError):
            tasks = IOTasksManager(self.get_configs())
            tasks.read(random.randint(1, 100), "Invalid type")

    def test_start_trigger(self):
        """
        Test to start a trigger source task
        """
        try:
            tasks = IOTasksManager(self.get_configs())
            tasks.start_trigger(
                "cDAQ1/segment1"
            )  #device name maybe different, also addd another test when device name is wrong
        except (Exception):
            self.fail("Could not start trigger")
        else:
            self.assertIsNotNone(tasks.activation_task.triggers)

    def start_trigger_fail_invalid_type(self):
        """
        AssertionError is raised if an invalid type for the trigger source is provided
        """
        tasks = IOTasksManager(self.get_configs())
        with self.assertRaises(AssertionError):
            tasks.start_trigger(100)
        with self.assertRaises(AssertionError):
            tasks.start_trigger({})
        with self.assertRaises(AssertionError):
            tasks.start_trigger([1, 2, 3, 4])

    def test_stop_tasks(self):
        """
        Test to close the tasks to and from the device
        """
        try:
            tasks = IOTasksManager(self.get_configs())
            tasks.stop_tasks()
        except (Exception):
            self.fail("Could not stop activation and readout tasks")

    def test_close_tasks(self):
        """
        Test to close the tasks to and from the device
        """
        try:
            tasks = IOTasksManager(self.get_configs())
            tasks.close_tasks()
        except (Exception):
            self.fail("Could not close activation and readout tasks")
        else:
            self.assertIsNotNone(tasks.readout_task)
            self.assertIsNotNone(tasks.activation_task)

    def test_write(self):
        """
        Test to write a random sample to the task
        """
        try:
            tasks = IOTasksManager(self.get_configs())
            tasks.write(np.random.rand(0, 100), False)
        except (Exception):
            self.fail("Could not close write data for these values")

    def test_write_fail_1(self):
        """
        Writing data raises an AssertionError if the input : y is not of type np.ndarray
        """
        tasks = IOTasksManager(self.get_configs())
        with self.assertRaises(AssertionError):
            tasks.write("invalid type", False)
        with self.assertRaises(AssertionError):
            tasks.write([1, 2, 3, 4], False)
        with self.assertRaises(AssertionError):
            tasks.write({}, False)

    def test_write_fail_2(self):
        """
        Writing data raises an AssertionError if the auto_start param is not a bool value
        """
        tasks = IOTasksManager(self.get_configs())
        with self.assertRaises(AssertionError):
            tasks.write(np.random.rand(3, 2), "invalid type")
        with self.assertRaises(AssertionError):
            tasks.write(np.random.rand(3, 2), [1, 2, 3, 4])
        with self.assertRaises(AssertionError):
            tasks.write(np.random.rand(3, 2), {})
        with self.assertRaises(AssertionError):
            tasks.write(np.random.rand(3, 2), 100)


if __name__ == "__main__":
    unittest.main()
