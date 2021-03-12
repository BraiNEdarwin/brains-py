""" The accelerator class enables to statically access the accelerator
(CUDA or CPU) that is used in the computer. The aim is to support both platforms seemlessly. """

import torch
from torch import nn

from brainspy.utils.manager import get_driver
from brainspy.utils.pytorch import TorchUtils
from brainspy.utils.waveform import WaveformManager

# deleteme
from brainspy.processors.simulation.processor import SurrogateModel


class HardwareProcessor(nn.Module):
    """
    The TorchModel class is used to manage together a torch model and its state dictionary. The usage is expected to be as follows
    mymodel = TorchModel()
    mymodel.load_model('my_path/my_model.pt')
    mymodel.model
    """

    # TODO: Automatically register the data type according to the configurations of the amplification variable of the  info dictionary

    def __init__(self, configs, logger=None):
        super(HardwareProcessor, self).__init__()
        self.driver = get_driver(configs)
        if configs["processor_type"] == "simulation_debug":
            self.voltage_ranges = self.driver.voltage_ranges
        else:
            self.voltage_ranges = TorchUtils.get_tensor_from_numpy(
                self.driver.voltage_ranges
            )
        self.waveform_mgr = WaveformManager(configs["data"]["waveform"])
        self.logger = logger
        # TODO: Manage amplification from this class
        self.amplification = configs["driver"]["amplification"]
        self.clipping_value = [
            configs["driver"]["output_clipping_range"][0] * self.amplification,
            configs["driver"]["output_clipping_range"][1] * self.amplification,
        ]

    def forward(self, x):
        with torch.no_grad():
            x, mask = self.waveform_mgr.plateaus_to_waveform(x, return_pytorch=False)
            output = self.forward_numpy(x)
            if self.logger is not None:
                self.logger.log_output(x)
        return TorchUtils.get_tensor_from_numpy(output[mask])

    def forward_numpy(self, x):
        return self.driver.forward_numpy(x)

    def reset(self):
        self.driver.reset()

    def close(self):
        if "close_tasks" in dir(self.driver):
            self.driver.close_tasks()
        else:
            print("Warning: Driver tasks have not been closed.")

    def is_hardware(self):
        return self.driver.is_hardware()
