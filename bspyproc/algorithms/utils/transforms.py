import numpy as np

from bspyproc.utils.pytorch import TorchUtils
from bspyproc.utils.waveform import WaveformManager
from bspyproc.utils.electrodes import get_map_to_voltage_vars


class DataToTensor():
    """Convert labelled data to pytorch tensor."""

    def __call__(self, data):
        inputs, targets = data[0], data[1]
        inputs = TorchUtils.get_tensor_from_numpy(inputs)
        targets = TorchUtils.get_tensor_from_numpy(targets)
        return (inputs, targets)


class DataToVoltageRange():

    def __init__(self, v_min, v_max, x_min=-1, x_max=1):
        self.scale, self.offset = get_map_to_voltage_vars(np.array(v_min), np.array(v_max), np.array(x_min), np.array(x_max))

    def __call__(self, data):
        inputs = data[0]
        inputs = (inputs * self.scale) + self.offset
        return (inputs, data[1])


class DataPointsToPlateau():

    def __init__(self, configs):
        self.mgr = WaveformManager(configs)

    def __call__(self, data):
        inputs, targets = data[0], data[1]

        inputs = self.mgr.points_to_plateaus(inputs)
        targets = self.mgr.points_to_plateaus(targets)

        return (inputs, targets)


class PointsToPlateau():

    def __init__(self, configs):
        self.mgr = WaveformManager(configs)

    def __call__(self, x):
        return self.mgr.points_to_plateaus(x)
