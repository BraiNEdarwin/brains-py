from email.policy import strict
import unittest

import torch
import numpy as np
import random
from brainspy.processors.modules import conv
from brainspy.utils.pytorch import TorchUtils
from brainspy.processors.processor import Processor


class ConvTest(unittest.TestCase):
    """
    Class for testing 'conv.py'.
    """
    def __init__(self, *args, **kwargs) -> None:
        super(ConvTest, self).__init__(*args, **kwargs)
        self.configs = {}
        self.configs["processor_type"] = "simulation"
        self.configs["electrode_effects"] = {}
        self.configs["electrode_effects"]["clipping_value"] = None
        self.configs["driver"] = {}
        self.configs["waveform"] = {}
        self.configs["waveform"]["plateau_length"] = 1
        self.configs["waveform"]["slope_length"] = 0

        self.info = {}
        self.info['model_structure'] = {}
        self.info['model_structure']['hidden_sizes'] = [90, 90, 90, 90, 90]
        self.info['model_structure']['D_in'] = 7
        self.info['model_structure']['D_out'] = 1
        self.info['model_structure']['activation'] = 'relu'
        self.info['electrode_info'] = {}
        self.info['electrode_info']['electrode_no'] = 8
        self.info['electrode_info']['activation_electrodes'] = {}
        self.info['electrode_info']['activation_electrodes'][
            'electrode_no'] = 7
        self.info['electrode_info']['activation_electrodes'][
            'voltage_ranges'] = np.array([[-0.55, 0.325], [-0.95, 0.55],
                                          [-1., 0.6], [-1., 0.6], [-1., 0.6],
                                          [-0.95, 0.55], [-0.55, 0.325]])
        self.info['electrode_info']['output_electrodes'] = {}
        self.info['electrode_info']['output_electrodes']['electrode_no'] = 1
        self.info['electrode_info']['output_electrodes']['amplification'] = [
            28.5
        ]
        self.info['electrode_info']['output_electrodes'][
            'clipping_value'] = None

        self.node = Processor(self.configs, self.info)
        self.input_indices = [[2, 3, 4]] * 3

    def test_init(self):
        try:
            c = conv.DNPUConv2d(
                self.node,
                data_input_indices=self.input_indices,
                in_channels=torch.randint(1, 20, (1, 1)).detach().cpu().item(),
                out_channels=torch.randint(1, 20,
                                           (1, 1)).detach().cpu().item(),
                kernel_size=3)
        except Exception:
            self.fail('Unable to initialise 3x3 DNPUConv')
        del c

        try:
            c = conv.DNPUConv2d(
                Processor(self.configs, self.info),
                data_input_indices=self.input_indices,
                in_channels=torch.randint(1, 20, (1, 1)).detach().cpu().item(),
                out_channels=torch.randint(1, 20,
                                           (1, 1)).detach().cpu().item(),
                kernel_size=3)
        except Exception:
            self.fail('Unable to initialise 3x3 DNPUConv from dict')
        del c

        try:
            c = conv.DNPUConv2d(
                self.node,
                data_input_indices=[[2, 3, 4, 5, 6]] * 5,
                in_channels=torch.randint(1, 20, (1, 1)).detach().cpu().item(),
                out_channels=torch.randint(1, 20,
                                           (1, 1)).detach().cpu().item(),
                kernel_size=5)
        except Exception:
            self.fail('Unable to initialise 5x5 DNPUConv')
        del c

    def test_conv(self):
        in_channels = torch.randint(1, 10, (1, 1)).detach().cpu().item()
        out_channels = in_channels = torch.randint(
            1, 10, (1, 1)).detach().cpu().item()
        kernel_size = torch.randint(3, 6, (1, 1)).detach().cpu().item()
        data_input_indices = torch.randint(0, 7, (kernel_size, kernel_size))
        for i in range(data_input_indices.shape[0]):
            while data_input_indices[i].unique(
            ).shape != data_input_indices[i].shape:
                data_input_indices[i] = torch.randint(
                    0, 7, (1, kernel_size)).squeeze()
        data_input_indices = data_input_indices.detach().cpu().tolist()
        stride = torch.randint(1, 3, (1, 1)).detach().cpu().item()
        padding = torch.randint(0, 5, (1, 1)).detach().cpu().item()
        batch_size = 2

        x = TorchUtils.format(torch.rand((batch_size, in_channels, 28, 28)))

        orig_conv = TorchUtils.format(
            torch.nn.Conv2d(in_channels,
                            out_channels,
                            kernel_size,
                            stride=stride,
                            padding=padding))
        res1 = orig_conv(x)
        del orig_conv
        c = TorchUtils.format(
            conv.DNPUConv2d(self.node,
                            data_input_indices=data_input_indices,
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding))
        c.add_input_transform([0, 1], strict=True)
        res2 = c(x)
        del c

    def test_conv_input_layer_unique(self):
        in_channels = torch.randint(1, 10, (1, 1)).detach().cpu().item()
        out_channels = in_channels = torch.randint(
                1, 10, (1, 1)).detach().cpu().item()
        kernel_size = 3
        data_input_indices = [[2, 3, 4]] * kernel_size
        batch_size = 2

        x = TorchUtils.format(torch.rand((batch_size, in_channels, 12, 12)))

        c = TorchUtils.format(
                conv.DNPUConv2d(self.node,
                                data_input_indices=data_input_indices,
                                in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size))
        c.add_input_transform([0, 1], strict=True)
        c(x)
        del c

            # input_indices = [[2, 3, 4]] * 3
            # c = TorchUtils.format(
            #     conv.DNPUConv2d(self.node,
            #                     data_input_indices=input_indices,
            #                     in_channels=in_channels,
            #                     out_channels=out_channels,
            #                     kernel_size=3,
            #                     stride=stride,
            #                     padding=padding))
            # c.add_input_transform([0, 1], strict=True)

            # del c
            # input_indices = [[0, 3, 4]] * 3
            # c = TorchUtils.format(
            #     conv.DNPUConv2d(self.node,
            #                     data_input_indices=input_indices,
            #                     in_channels=in_channels,
            #                     out_channels=out_channels,
            #                     kernel_size=3,
            #                     stride=stride,
            #                     padding=padding))
            # c.add_input_transform([0, 1], strict=True)
            # del c


if __name__ == "__main__":
    unittest.main()