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
        self.info['electrode_info']['activation_electrodes']['electrode_no'] = 7
        self.info['electrode_info']['activation_electrodes'][
            'voltage_ranges'] = np.array([[-0.55, 0.325], [-0.95, 0.55],
                                          [-1., 0.6], [-1., 0.6], [-1., 0.6],
                                          [-0.95, 0.55], [-0.55, 0.325]])
        self.info['electrode_info']['output_electrodes'] = {}
        self.info['electrode_info']['output_electrodes']['electrode_no'] = 1
        self.info['electrode_info']['output_electrodes']['amplification'] = [28.5]
        self.info['electrode_info']['output_electrodes']['clipping_value'] = None

        self.node = Processor(self.configs, self.info)
        self.conv = conv.DNPUConv2d(self.node, data_input_indices=[[3, 4]],
                                    in_channels=3, out_channels=5, kernel_size=5)


if __name__ == "__main__":
    unittest.main()