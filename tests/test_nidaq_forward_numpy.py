import unittest
import numpy as np
import random
import brainspy
from brainspy.processors.hardware.drivers.ni.setup import NationalInstrumentsSetup
from utils import check_test_configs
from brainspy.processors.hardware.drivers.nidaq import CDAQtoNiDAQ


class NIDAQ_Forward_Numpy_Test(unittest.TestCase):
    """
    Test forward_numpy() of the Nidaq Driver.

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
    def test_forward_numpy_simple(self):
        """
        A simple test for the forward numpy method with a helper method
        """
        configs = self.get_configs()
        nidaq = CDAQtoNiDAQ(configs)

        point_no = 1000
        vmax = 0.08
        vmin = -0.08
        input_electrode = 5
        activation_electrode_no = len(
            nidaq.configs['instruments_setup']['activation_channels'])
        data = np.zeros((activation_electrode_no, point_no))
        data[input_electrode] = self.generate_sample(vmax,
                                                     vmin,
                                                     point_no,
                                                     up_direction=True)
        print(data.shape)
        try:
            nidaq.forward_numpy(data.T)
        except (Exception):
            self.fail("Could not make forward pass")
        finally:
            nidaq.close_tasks()

    @unittest.skipUnless(
        brainspy.__TEST_MODE__ == "HARDWARE_NIDAQ",
        "Method deactivated as it is only possible to be tested on a CDAQ TO NIDAQ setup"
    )
    def generate_sample(self,
                        v_low: float,
                        v_high: float,
                        point_no: int,
                        up_direction: bool = False):
        """
        THis is a helper funtion for the above test
        Geneartes the sample data for the input electrode data
        """
        assert point_no % 2 == 0, 'Only an even point number is accepted.'
        point_no = int(point_no / 2)

        if up_direction:
            aux = v_low
            v_low = v_high
            v_high = aux

        ramp1 = np.linspace(0, v_low,
                            round((point_no * v_low) / (v_low - v_high)))
        ramp2 = np.linspace(v_low, v_high, point_no)
        ramp3 = np.linspace(v_high, 0,
                            round((point_no * v_high) / (v_high - v_low)))

        result = np.concatenate((ramp1, ramp2, ramp3))
        return result

    @unittest.skipUnless(
        brainspy.__TEST_MODE__ == "HARDWARE_NIDAQ",
        "Method deactivated as it is only possible to be tested on a CDAQ TO NIDAQ setup"
    )
    def test_forward_numpy_random(self):
        """
        Test the forward pass with random input data
        """
        configs = self.get_configs()
        a1 = random.randint(1, 1000)
        a2 = len(configs["instruments_setup"]["activation_channels"])

        nidaq = CDAQtoNiDAQ(configs)
        y = np.random.rand(a1, a2) / 1000
        # Force them to start and end at zero
        y[0] = np.zeros_like(a2)
        y[-1] = np.zeros_like(a2)
        try:
            nidaq.original_shape = y.shape[0]
            val = nidaq.forward_numpy(y)
        except (Exception):
            self.fail("Could not synchronise output data")
        finally:
            nidaq.close_tasks()
            self.assertIsNotNone(val)

    @unittest.skipUnless(
        brainspy.__TEST_MODE__ == "HARDWARE_NIDAQ",
        "Method deactivated as it is only possible to be tested on a CDAQ TO NIDAQ setup"
    )
    def test_forward_numpy_invalid_type(self):
        """
        Invalid type for input raises an AssertionError
        """
        configs = self.get_configs()
        nidaq = CDAQtoNiDAQ(configs)
        with self.assertRaises(AssertionError):
            nidaq.forward_numpy("Invalid type")
        with self.assertRaises(AssertionError):
            nidaq.forward_numpy(500)
        with self.assertRaises(AssertionError):
            nidaq.forward_numpy(100.10)
        with self.assertRaises(AssertionError):
            nidaq.forward_numpy({"dict_key": 2})
        with self.assertRaises(AssertionError):
            nidaq.forward_numpy([1, 2, 3, 4, 5, 6])
        nidaq.close_tasks()


if __name__ == "__main__":
    testobj = NIDAQ_Forward_Numpy_Test()
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
