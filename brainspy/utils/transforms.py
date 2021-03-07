import torch
import numpy as np

from brainspy.utils.pytorch import TorchUtils
from brainspy.utils.waveform import WaveformManager
from brainspy.utils.electrodes import transform_current_to_voltage


# class CurrentToVoltage():
#     def __init__(self, v_min, v_max, x_min=-1, x_max=1, cut=True):
#         self.scale, self.offset = get_map_to_voltage_vars(v_min, v_max, x_min, x_max)
#         self.x_min = x_min
#         self.x_max = x_max
#         self.cut = cut

#     def __call__(self, x):
#         if self.cut:
#             x = torch.clamp(x, min=self.x_min, max=self.x_max)
#         x = (x * self.scale) + self.offset
#         return x


class CurrentToVoltage:
    def __init__(self, current_range, voltage_range, cut=True):
        assert len(current_range) == len(
            voltage_range
        ), "Mapping ranges are different in length"
        self.map_variables = TorchUtils.get_tensor_from_list(
            [
                transform_current_to_voltage(
                    voltage_range[i][0],
                    voltage_range[i][1],
                    current_range[i][0],
                    current_range[i][1],
                )
                for i in range(len(current_range))
            ]
        )
        self.current_range = current_range
        self.cut = cut

    def __call__(self, x):
        aux1 = x.clone()
        aux2 = torch.zeros_like(x)
        assert len(x.shape) == 2 and x.shape[1] == len(
            self.map_variables
        ), "Input shape not supported."
        for i in range(len(self.map_variables)):
            # Linear transformation variables are as follows
            # SCALE: self.map_variables[i][0]
            # OFFSET: self.map_variables[i][1]
            if self.cut:
                aux1[:, i] = torch.clamp(
                    x[:, i], min=self.current_range[i][0], max=self.current_range[i][1]
                )
            aux2[:, i] = (aux1[:, i] * self.map_variables[i][0]) + self.map_variables[
                i
            ][1]
        x = aux2.clone()
        del aux1
        del aux2
        return x


class MinMaxScaler:
    def __init__(self):
        pass

    def __call__(self, x):
        assert len(x.shape) == 2, "Only two dimensional tensors supported"
        aux = x.clone()
        for i in range(x.shape[1]):
            aux[:, i] = (x[:, i] - x[:, i].min()) / (x[:, i].max() - x[:, i].min())
        x = aux.clone()
        return x


class DataToTensor:
    """Convert labelled data to pytorch tensor."""

    def __init__(self, device=None):
        if device is not None:
            self.device = device
        else:
            self.device = TorchUtils.get_accelerator_type()

    def __call__(self, data):
        inputs, targets = data[0], data[1]
        inputs = torch.tensor(
            inputs, device=self.device, dtype=TorchUtils.get_data_type()
        )
        targets = torch.tensor(
            targets, device=self.device, dtype=TorchUtils.get_data_type()
        )
        return (inputs, targets)


class ToDevice:
    """Inputs and targets are transferred to GPU if necessary"""

    def __call__(self, data):
        inputs, targets = data[0], data[1]
        if inputs.device != TorchUtils.get_accelerator_type():
            inputs = inputs.to(device=TorchUtils.get_accelerator_type())
        if targets.device != TorchUtils.get_accelerator_type():
            targets = targets.to(device=TorchUtils.get_accelerator_type())
        return (inputs, targets)


class DataToVoltageRange:
    def __init__(self, v_min, v_max, x_min=-1, x_max=1):
        self.scale, self.offset = transform_current_to_voltage(
            np.array(v_min), np.array(v_max), np.array(x_min), np.array(x_max)
        )

    def __call__(self, data):
        inputs = data[0]
        inputs = (inputs * self.scale) + self.offset
        return (inputs, data[1])


class DataPointsToPlateau:
    def __init__(self, configs):
        self.mgr = WaveformManager(configs)

    def __call__(self, data):
        inputs, targets = data[0], data[1]

        inputs = self.mgr.points_to_plateaus(inputs)
        targets = self.mgr.points_to_plateaus(targets)

        return (inputs, targets)


class PlateausToPoints:
    def __init__(self, configs):
        self.mgr = WaveformManager(configs)

    def __call__(self, x):
        return self.mgr.plateaus_to_points(x)


class PointsToPlateaus:
    def __init__(self, configs):
        self.mgr = WaveformManager(configs)

    def __call__(self, x):
        return self.mgr.points_to_plateaus(x)
