""" The accelerator class enables to statically access the accelerator
(CUDA or CPU) that is used in the computer. The aim is to support both platforms seemlessly. """

import torch
from torch import nn

from bspyproc.processors.hardware.drivers.driver_mgr import get_driver
from bspyproc.utils.pytorch import TorchUtils
from bspyproc.utils.waveform import WaveformManager

# deleteme
from bspyproc.processors.simulation.surrogate import SurrogateModel


class HardwareProcessor(nn.Module):
    """
        The TorchModel class is used to manage together a torch model and its state dictionary. The usage is expected to be as follows
        mymodel = TorchModel()
        mymodel.load_model('my_path/my_model.pt')
        mymodel.model
    """
# TODO: Automatically register the data type according to the configurations of the amplification variable of the  info dictionary

    def __init__(self, configs, logger=None):
        super().__init__()
        self._init_voltage_range(configs)
        self.driver = get_driver(configs)
        self.waveform_mgr = WaveformManager(configs['waveform'])
        self.logger = logger
        # TODO: Manage amplification from this class
        self.amplification = configs['amplification']

    def _init_voltage_range(self, configs):
        offset = TorchUtils.get_tensor_from_list(configs['offset'])
        amplitude = TorchUtils.get_tensor_from_list(configs['amplitude'])
        self.min_voltage = offset - amplitude
        self.max_voltage = offset + amplitude

    def forward(self, x):
        with torch.no_grad():
            x = TorchUtils.get_numpy_from_tensor(x)
            x, mask = self.waveform_mgr.plateaus_to_waveform(x)
            output = self.forward_numpy(x)
            if self.logger is not None:
                self.logger.log_output(x)
        return TorchUtils.get_tensor_from_numpy(output[mask])

    def forward_numpy(self, x):
        return self.driver.forward_numpy(x)

    def reset(self):
        self.driver.reset()

    def close(self):
        self.driver.close_tasks()
