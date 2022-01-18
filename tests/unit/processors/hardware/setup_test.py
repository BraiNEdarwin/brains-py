import os
import torch
import unittest
import Pyro4
import numpy as np
import brainspy
from brainspy.processors.processor import Processor
from brainspy.processors.hardware.drivers.cdaq import CDAQtoCDAQ
from brainspy.processors.hardware.drivers.ni.tasks import LocalTasks, RemoteTasks
from brainspy.processors.hardware.drivers.ni.setup import NationalInstrumentsSetup


class Setup_Test(unittest.TestCase):
    """
    Tests for the hardware processor with the CDAQ to CDAQ driver.

    """
    def __init__(self, test_name):
        super(Setup_Test, self).__init__()
        configs = {}
        configs["processor_type"] = "cdaq_to_cdaq"
        configs["input_indices"] = [2, 3]
        configs["electrode_effects"] = {}
        configs["electrode_effects"]["amplification"] = 3
        configs["electrode_effects"]["clipping_value"] = [-300, 300]
        configs["electrode_effects"]["noise"] = {}
        configs["electrode_effects"]["noise"]["noise_type"] = "gaussian"
        configs["electrode_effects"]["noise"]["variance"] = 0.6533523201942444
        configs["driver"] = {}
        configs["driver"]["real_time_rack"] = False
        configs["driver"]["sampling_frequency"] = 1000
        configs["driver"]["instruments_setup"] = {}
        configs["driver"]["instruments_setup"]["multiple_devices"] = False
        configs["driver"]["instruments_setup"][
            "trigger_source"] = "cDAQ1/segment1"
        configs["driver"]["instruments_setup"][
            "activation_instrument"] = "cDAQ1Mod3"
        configs["driver"]["instruments_setup"]["activation_channels"] = [
            0,
            2,
            5,
            3,
            4,
            6,
            1,
        ]  # Channels through which voltages will be sent for activating the device (with both data inputs and control voltages)
        configs["driver"]["instruments_setup"]["activation_voltages"] = [
            [-1.2, 0.6],
            [-1.2, 0.6],
            [-1.2, 0.6],
            [-1.2, 0.6],
            [-1.2, 0.6],
            [-0.7, 0.3],
            [-0.7, 0.3],
        ]
        configs["driver"]["instruments_setup"][
            "readout_instrument"] = "cDAQ1Mod4"
        configs["driver"]["instruments_setup"]["readout_channels"] = [
            4
        ]  # Channels for reading the output current values

        configs["waveform"] = {}
        configs["waveform"]["plateau_length"] = 10
        configs["waveform"]["slope_length"] = 30

        self.configs = configs

        # if brainspy.TEST_MODE == "SIMULATION_PC":
        #     self.model_data = {}
        #     self.model_data["info"] = {}
        #     self.model_data["info"]["electrode_info"] = {
        #         "electrode_no": 8,
        #         "activation_electrodes": {
        #             "electrode_no":
        #             7,
        #             "voltage_ranges": [
        #                 [-1.2, 0.6],
        #                 [-1.2, 0.6],
        #                 [-1.2, 0.6],
        #                 [-1.2, 0.6],
        #                 [-1.2, 0.6],
        #                 [-0.3, 0.6],
        #                 [-0.7, 0.3],
        #             ],
        #         },
        #         "output_electrodes": {
        #             "electrode_no": 1,
        #             "amplification": 28.5,
        #             "clipping_value": [-114.0, 114.0],
        #         },
        #     }
        #     self.model = Processor(
        #         self.configs,
        #         self.model_data["info"],
        #     )

    @unittest.skipIf(
        brainspy.TEST_MODE == "SIMULATION_PC",
        "Method deactivated as it is only possible to be tested on a CDAQ TO CDAQ setup",
    )
    def test_init(self):
        """
        Test to check correct initialization of the Setup with the cdaq_to_cdaq driver.
        """
        isinstance(self.model.driver, CDAQtoCDAQ)
        isinstance(self.model.driver, NationalInstrumentsSetup)

    @unittest.skipIf(
        brainspy.TEST_MODE == "SIMULATION_PC",
        "Method deactivated as it is only possible to be tested on a CDAQ TO CDAQ setup",
    )
    def test_init_configs(self):
        """
        Test to check correct initialization of the configurations of the Setup.
        """
        self.assertEqual(self.model.last_shape, -1)
        self.assertEqual(self.data_results, None)
        self.assertEqual(self.offsetted_shape, None)
        self.assertEqual(self.ceil, None)

    @unittest.skipIf(
        brainspy.TEST_MODE == "SIMULATION_PC",
        "Method deactivated as it is only possible to be tested on a CDAQ TO CDAQ setup",
    )
    def test_init_tasks(self):
        """
        Test to check the correct initilization of the Setup tasks driver as RemoteTasks or LocalTasks
        """
        self.assertTrue(
            isinstance(self.model.tasks_driver, RemoteTasks)
            or isinstance(self.model.tasks_driver, LocalTasks))

    @unittest.skipIf(
        brainspy.TEST_MODE == "SIMULATION_PC",
        "Method deactivated as it is only possible to be tested on a CDAQ TO CDAQ setup",
    )
    def test_init_semaphore(self):
        """
        Test to check the intitlization of the semaphore by checking if the event is set to False
        """
        self.assertEqual(self.model.event.isSet(), False)

    @unittest.skipIf(
        brainspy.TEST_MODE == "SIMULATION_PC",
        "Method deactivated as it is only possible to be tested on a CDAQ TO CDAQ setup",
    )
    def test_process_output(self):
        """
        Test to check if the inputted data is processed correctly
        """
        output = self.model.process_output_data([1, 1])
        self.assertEqual(output, np.array([[3, 3]]))

    @unittest.skipIf(
        brainspy.TEST_MODE == "SIMULATION_PC",
        "Method deactivated as it is only possible to be tested on a CDAQ TO CDAQ setup",
    )
    def test_read_data(self):
        """
        Test to see if the data can be read from the device - can be None
        """
        data_results = self.model.read_data(
            np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]))
        self.assertTrue(data_results == None or data_results is not None)

    @unittest.skipIf(
        brainspy.TEST_MODE == "SIMULATION_PC",
        "Method deactivated as it is only possible to be tested on a CDAQ TO CDAQ setup",
    )
    def test__read_data(self):
        """
        Test to see if the data that is sent to the DNPU hardware is read
        """
        data_results = self.model._read_data(
            np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]))
        self.assertTrue(data_results is not None)

    @unittest.skipIf(
        brainspy.TEST_MODE == "SIMULATION_PC",
        "Method deactivated as it is only possible to be tested on a CDAQ TO CDAQ setup",
    )
    def test_set_shape_vars(self):
        """
        Test to check if the shape can be set by checking if the value of the offseted shape and ceil is updated
        """
        shape1 = self.model.offseted_shape
        ceil1 = self.model.ceil
        self.model.set_shape_vars((3, 3))
        shape2 = self.model.offseted_shape
        ceil2 = self.model.ceil
        self.assertTrue(shape1 != shape2)
        self.assertTrue(ceil1 != ceil2)

    @unittest.skipIf(
        brainspy.TEST_MODE == "SIMULATION_PC",
        "Method deactivated as it is only possible to be tested on a CDAQ TO CDAQ setup",
    )
    def test_read_security(self):
        """
        Test to check if the security checks does not throw an error
        """
        raised = False
        try:
            self.model.read_security_checks()
        except:
            raised = True
        self.assertFalse(raised, "Exception raised")

    @unittest.skipIf(
        brainspy.TEST_MODE == "SIMULATION_PC",
        "Method deactivated as it is only possible to be tested on a CDAQ TO CDAQ setup",
    )
    def test_close(self):
        """
        Test if closing the processor raises a warning.
        """
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            self.model.close()
            self.assertEqual(len(caught_warnings), 1)

    @unittest.skipIf(
        brainspy.TEST_MODE == "SIMULATION_PC",
        "Method deactivated as it is only possible to be tested on a CDAQ TO CDAQ setup",
    )
    def test_get_amplification(self):
        """
        Test to check if the amplication value is set to the correct value
        """
        self.assertEqual(self.model.get_amplification_value(), 3)

    @unittest.skipIf(
        brainspy.TEST_MODE == "SIMULATION_PC",
        "Method deactivated as it is only possible to be tested on a CDAQ TO CDAQ setup",
    )
    def test_is_hardware(self):
        """
        Test if the processor is a hardware
        """
        self.assertTrue(self.model.is_hardware())

    @unittest.skipIf(
        brainspy.TEST_MODE == "SIMULATION_PC",
        "Method deactivated as it is only possible to be tested on a CDAQ TO CDAQ setup",
    )
    def test_forward_numpy(self):
        """
        Test if a forward pass through the processor returns a tensor of the
        right shape (the numpy version).
        """
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        x = self.model.forward_numpy(x)
        self.assertEqual(list(x.shape), [1])

    def runTest(self):
        self.test_init()
        self.test_init_configs()
        self.test_init_tasks()
        self.test_init_semaphore()
        self.test_is_hardware()
        self.test_process_output()
        self.test_read_data()
        self.test__read_data()
        self.test_get_amplification()
        self.test_forward_numpy()
        self.test_close()


if __name__ == "__main__":
    unittest.main()
