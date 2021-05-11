import os
import torch
import unittest
import numpy as np
import warnings
from brainspy.utils.pytorch import TorchUtils
from brainspy.processors.simulation.processor import SurrogateModel
from brainspy.utils.waveform import WaveformManager
from brainspy.processors.hardware.processor import HardwareProcessor


class Hardware_Processor_Test(unittest.TestCase):

    """
    Tests for the hardware processor in "simulation_debug" mode.

    """

    def __init__(self, test_name):
        super(Hardware_Processor_Test, self).__init__()

        configs = {}
        configs["processor_type"] = "simulation_debug"
        configs["input_indices"] = [2, 3]
        configs["electrode_effects"] = {}
        configs["electrode_effects"]["amplification"] = 28.5
        configs["electrode_effects"]["clipping_value"] = [-110, 110]
        configs["electrode_effects"]["noise"] = {}
        configs["electrode_effects"]["noise"]["noise_type"] = "gaussian"
        configs["electrode_effects"]["noise"]["variance"] = 0.6533523201942444
        configs["waveform"] = {}
        configs["waveform"]["plateau_length"] = 10
        configs["waveform"]["slope_length"] = 30

        self.configs = configs
        while "brains-py" not in os.getcwd():
            os.chdir("..")
            os.chdir("brains-py")

        self.model_dir = os.path.join(
            os.getcwd(), "tests/unit/utils/testfiles/training_data.pt"
        )
        self.model_data = torch.load(self.model_dir, map_location=torch.device("cpu"))

        self.debug_model = SurrogateModel(
            self.model_data["info"]["model_structure"],
            self.model_data["model_state_dict"],
        )
        self.model = HardwareProcessor(
            self.debug_model,
            slope_length=self.configs["waveform"]["slope_length"],
            plateau_length=self.configs["waveform"]["plateau_length"],
        )

    def test_init(self):
        """
        Test to check correct initialization of the Hardware processor in simulation debug model.
        Hardware processor is initialized as an instance of the Surrogate Model.
        """

        isinstance(self.model.driver, SurrogateModel)

    def test_forward(self):
        """
        Test if a forward pass through the processor returns a tensor of the
        right shape.
        """
        data = [
            [1.0, 14.0, 5.0, 10.0, 9.0, 6.0, 13.0],
            [2.0, 15.0, 6.0, 11.0, 10.0, 7.0, 14.0],
            [3.0, 16.0, 7.0, 12.0, 11.0, 8.0, 15.0],
            [4.0, 17.0, 8.0, 13.0, 12.0, 9.0, 16.0],
        ]
        data = TorchUtils.format(data)
        mgr = WaveformManager(self.configs["waveform"])
        data_plateaus = mgr.points_to_plateaus(data)
        x = self.model.forward(data_plateaus)
        self.assertEqual(list(x.shape), [40, 1])

    def test_forward_numpy(self):
        """
        Test if a forward pass through the processor returns a tensor of the
        right shape (the numpy version).
        """
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        x = self.model.forward_numpy(x)
        self.assertEqual(list(x.shape), [1])

    def test_close(self):
        """
        Test if closing the processor raises a warning.
        """
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            self.model.close()
            self.assertEqual(len(caught_warnings), 1)

    def test_is_hardware(self):
        """
        Test if the processor is a hardware,but in this case is an instance of a Surrogate Model.
        """
        self.assertFalse(self.model.is_hardware())

    def runTest(self):
        self.test_init()
        self.test_forward()
        self.test_forward_numpy()
        self.test_close()
        self.test_is_hardware()


if __name__ == "__main__":
    unittest.main()
