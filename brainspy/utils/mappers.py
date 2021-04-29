import torch
from torch import nn
from brainspy.utils.pytorch import TorchUtils
from brainspy.utils.transforms import get_linear_transform_constants
from torch.quantization import PerChannelMinMaxObserver


class SimpleMapping:
    def __init__(self, input_range, output_range, clip_input=True):
        input_range = self.init_input_range(input_range, output_range)
        self.amplitude, self.offset = get_linear_transform_constants(
            output_range[0], output_range[1], input_range[0], input_range[1]
        )

    def __call__(self, x):
        return (self.amplitude * x) + self.offset

    def init_input_range(self, input_range, output_range):
        min_input = input_range[0]
        max_input = input_range[1]
        input_range = torch.ones_like(output_range)
        input_range[0] *= min_input
        input_range[1] *= max_input
        return input_range


class VariableRangeMapper(nn.Module):
    def __init__(self, output_range, averaging_constant=0.8):
        super(VariableRangeMapper, self).__init__()

        self.min_range = output_range[:, 0]
        self.max_range = output_range[:, 1]
        self.out_min = TorchUtils.format_tensor(nn.Parameter(self.min_range.clone()))
        self.out_max = TorchUtils.format_tensor(nn.Parameter(self.max_range.clone()))
        self.observer = ExponentialAveragePerChannelMinMaxObserver(
            averaging_constant=averaging_constant, ch_axis=1
        )
        # self.observer = torch.quantization.PerChannelMinMaxObserver(1)
        # self.observer = torch.quantization.MovingAveragePerChannelMinMaxObserver(ch_axis=1)

    def get_scale(self):
        v = self.out_min - self.out_max
        x = self.observer.min_vals - self.observer.max_vals
        return v / x

    def get_offset(self):
        x = self.observer.min_vals - self.observer.max_vals
        v = (self.out_max * self.observer.min_vals) - (
            self.out_min * self.observer.max_vals
        )
        return v / x

    def forward(self, x):
        self.observer(x)
        x = (x * self.get_scale()) + self.get_offset()
        return x

    def regularizer(self):
        return torch.sum(
            torch.relu(self.min_range - self.out_min) + torch.relu(self.out_max - self.max_range)
        )


class STDRangeMapper(nn.Module):
    def __init__(self, output_range, std_times=2):
        super(STDRangeMapper, self).__init__()

        self.min_range = output_range[:, 0]
        self.max_range = output_range[:, 1]
        self.out_min = TorchUtils.format_tensor(nn.Parameter(self.min_range.clone()))
        self.out_max = TorchUtils.format_tensor(nn.Parameter(self.max_range.clone()))
        self.std_times = std_times
        # self.observer = ExponentialAveragePerChannelMinMaxObserver(averaging_constant=averaging_constant, ch_axis=1)
        # self.observer = torch.quantization.PerChannelMinMaxObserver(1)
        # self.observer = torch.quantization.MovingAveragePerChannelMinMaxObserver(ch_axis=1)

    def update(self, bn):
        std = torch.sqrt(bn.running_var + bn.eps) * self.std_times
        self.min_vals = -std
        self.max_vals = std

    def get_scale(self):
        v = self.out_min - self.out_max
        x = self.min_vals - self.max_vals
        return v / x

    def get_offset(self):
        x = self.min_vals - self.max_vals
        v = (self.out_max * self.min_vals) - (self.out_min * self.max_vals)
        return v / x

    def forward(self, x):
        # self.observer(x)
        z = torch.zeros_like(x)
        for i in range(x.shape[-1]):
            z[:, i] = torch.clamp(x[:, i], min=self.min_vals[i], max=self.max_vals[i])
        z = (z * self.get_scale()) + self.get_offset()
        return z

    def regularizer(self):
        return torch.sum(
            torch.relu(self.min_range - self.out_min) + torch.relu(self.out_max - self.max_range)
        )


class ExponentialAveragePerChannelMinMaxObserver(PerChannelMinMaxObserver):
    def __init__(
        self,
        averaging_constant=0.9,
        ch_axis=0,
        dtype=torch.quint8,
        qscheme=torch.per_channel_affine,
        reduce_range=False,
        quant_min=None,
        quant_max=None,
    ):
        super(ExponentialAveragePerChannelMinMaxObserver, self).__init__(
            ch_axis=ch_axis,
            dtype=dtype,
            qscheme=qscheme,
            reduce_range=reduce_range,
            quant_min=quant_min,
            quant_max=quant_max,
        )
        self.averaging_constant = averaging_constant
        self.averaging_constant_opposite = 1 - averaging_constant

    def forward(self, x_orig):
        x = x_orig.detach()  # avoid keeping autograd tape
        x = x.to(self.min_vals.dtype)
        min_vals = self.min_vals
        max_vals = self.max_vals
        x_dim = x.size()

        new_axis_list = [i for i in range(len(x_dim))]  # noqa: C416
        new_axis_list[self.ch_axis] = 0
        new_axis_list[0] = self.ch_axis
        y = x.permute(new_axis_list)
        y = torch.flatten(y, start_dim=1)
        if min_vals.numel() == 0 or max_vals.numel() == 0:
            min_vals, max_vals = torch._aminmax(y, 1)
        else:
            min_vals_cur, max_vals_cur = torch._aminmax(y, 1)
            min_vals = (min_vals * self.averaging_constant) + (
                self.averaging_constant_opposite * min_vals_cur
            )
            max_vals = (max_vals * self.averaging_constant) + (
                self.averaging_constant_opposite * max_vals_cur
            )
        self.min_vals.resize_(min_vals.shape)
        self.max_vals.resize_(max_vals.shape)
        self.min_vals.copy_(min_vals)
        self.max_vals.copy_(max_vals)
        return x_orig
