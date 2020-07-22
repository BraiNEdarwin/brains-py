""" The accelerator class enables to statically access the accelerator
(CUDA or CPU) that is used in the computer. The aim is to support both platforms seemlessly. """

import torch
import torch.nn as nn

from bspyproc.processors.hardware.drivers.driver_mgr import get_driver
from bspyproc.utils.pytorch import TorchUtils
from bspyproc.utils.waveform import WaveformManager


class HardwareProcessor(nn.Module):
    """
        The TorchModel class is used to manage together a torch model and its state dictionary. The usage is expected to be as follows
        mymodel = TorchModel()
        mymodel.load_model('my_path/my_model.pt')
        mymodel.model
    """
# TODO: Automatically register the data type according to the configurations of the amplification variable of the  info dictionary

    def __init__(self, configs):
        super().__init__()
        self.load_model(configs)
        self._init_voltage_range()
        self.driver = get_driver(configs)
        self.waveform_mgr = WaveformManager(configs)
        # TODO: Manage amplification from this class
        self.amplification = self.info['data_info']['processor']['amplification']

    def _init_voltage_range(self):
        offset = TorchUtils.get_tensor_from_list(self.info['data_info']['input_data']['offset'])
        amplitude = TorchUtils.get_tensor_from_list(self.info['data_info']['input_data']['amplitude'])
        self.min_voltage = offset - amplitude
        self.max_voltage = offset + amplitude

    def forward(self, x):
        with torch.no_grad():
            inputs_numpy = TorchUtils.get_numpy_from_tensor(x)
            output = self.forward_numpy(inputs_numpy)
        return TorchUtils.get_tensor_from_numpy(output)

    def forward_numpy(self, x):
        return self.driver.get_output(x)

    def reset(self):
        print("Warning: Reset function in Surrogate Model not implemented.")
        self.driver.reset()

    def close(self):
        pass
