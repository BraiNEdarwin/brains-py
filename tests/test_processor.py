import unittest
import torch
import brainspy
from brainspy.utils.pytorch import TorchUtils
from brainspy.processors.processor import Processor
from utils import get_configs, get_custom_model_configs


class Processor_Test_CDAQ(unittest.TestCase):
    """
    Tests for the hardware processor with a CDAQ driver.

    To run this file, the device has to be connected to a CDAQ setup and
    the device configurations have to be specified depending on the setup.

    The test mode has to be set to HARDWARE_CDAQ in tests/main.py.
    The required keys have to be defined in the get_configs_CDAQ() function.

    Some sample keys have been defined to run tests which do not require connection
    to the hardware.

    """

    # def get_processor_configs(self):
    #     """
    #     Get the configs to initialize the hardware processor
    #     """
    #     configs = {}
    #     configs["waveform"] = {}
    #     configs["waveform"]["plateau_length"] = 10
    #     configs["waveform"]["slope_length"] = 30

    #     configs["amplification"] = 100
    #     configs["inverted_output"] = True
    #     configs["output_clipping_range"] = [-1, 1]

    #     configs["instrument_type"] = "cdaq_to_cdaq"

    #     configs["instruments_setup"] = {}

    #     configs["instruments_setup"]["multiple_devices"] = False
    #     # TODO Specify the name of the Trigger Source
    #     configs["instruments_setup"]["trigger_source"] = "a"

    #     # TODO Specify the name of the Activation instrument
    #     configs["instruments_setup"]["activation_instrument"] = "b"

    #     # TODO Specify the Activation channels (pin numbers)
    #     # For example, [1,2,3,4,5,6,7]
    #     configs["instruments_setup"]["activation_channels"] = [
    #         1, 2, 3, 4, 5, 6, 7
    #     ]

    #     # TODO Specify the activation Voltage ranges
    #     # For example, [[-1.2, 0.6],[-1.2, 0.6],[-1.2, 0.6],[-1.2, 0.6],[-1.2, 0.6],[-0.7, 0.3],[-0.7, 0.3]]
    #     configs["instruments_setup"]["activation_voltage_ranges"] = [
    #         [-1.2, 0.6], [-1.2, 0.6], [-1.2, 0.6], [-1.2, 0.6], [-1.2, 0.6],
    #         [-0.7, 0.3], [-0.7, 0.3]
    #     ]

    #     # TODO Specify the name of the Readout Instrument
    #     configs["instruments_setup"]["readout_instrument"] = "c"

    #     # TODO Specify the readout channels
    #     # For example, [4]
    #     configs["instruments_setup"]["readout_channels"] = [4]
    #     configs["instruments_setup"]["activation_sampling_frequency"] = 500
    #     configs["instruments_setup"]["readout_sampling_frequency"] = 1000
    #     configs["instruments_setup"]["average_io_point_difference"] = True

    #     return configs

    @unittest.skipUnless(brainspy.__TEST_MODE__ == "HARDWARE_CDAQ",
                         "Hardware test is skipped for simulation setup.")
    def test_init_simulation(self):
        """
        Test to check correct initialization of the Hardware processor.
        """
        configs, model_data = get_custom_model_configs()
        state_dict = torch.load('tests/data/random_state_dict.pt')
        try:
            model = None
            model = Processor(
                configs,
                model_data['info'],
                state_dict,
            )
        except (Exception):
            if model is not None:
                model.close()
            self.fail("Could not initialize processor")

    @unittest.skipUnless(brainspy.__TEST_MODE__ == "HARDWARE_CDAQ",
                         "Hardware test is skipped for simulation setup.")
    def test_init_cdaq(self):
        driver_configs = get_configs()
        processor_configs, model_data = get_custom_model_configs()
        processor_configs["processor_type"] = 'cdaq_to_cdaq'
        processor_configs['driver'] = driver_configs
        #del processor_configs['driver']['amplification']
        del processor_configs["driver"]["instruments_setup"][
            "activation_voltage_ranges"]
        try:
            model = None
            model = Processor(processor_configs, model_data['info'])
            model.get_readout_electrode_no()
        except (Exception):
            if model is not None:
                model.close()
            self.fail("Could not initialize processor")

    @unittest.skipUnless(brainspy.__TEST_MODE__ == "HARDWARE_CDAQ",
                         "Hardware test is skipped for simulation setup.")
    def test_init_cdaq_no_electrode_effects(self):
        processor_configs, model_data = get_custom_model_configs()
        processor_configs["processor_type"] = 'simulation_debug'
        try:
            model = None
            model = Processor(processor_configs, model_data['info'])
            model.get_readout_electrode_no()
        except (Exception):
            if model is not None:
                model.close()
            self.fail("Could not initialize processor")

    def test_init_fail(self):
        processor_configs, model_data = get_custom_model_configs()
        processor_configs["processor_type"] = 'asdf'
        with self.assertRaises(NotImplementedError):
            model = None
            model = Processor(processor_configs, model_data['info'])
        if model is not None:
            model.close()

    def test_format_targets(self):
        """
        Test to check correct initialization of the Hardware processor.
        """
        processor_configs, model_data = get_custom_model_configs()
        try:
            model = None
            model = Processor(processor_configs,
                              model_data['info'],
                              average_plateaus=False)
            model = TorchUtils.format(model)
            res = model.format_targets(TorchUtils.format(torch.rand((3, 7))))
        except (Exception):
            if model is not None:
                model.close()
            self.fail("Could not initialize processor")


if __name__ == "__main__":
    unittest.main()
