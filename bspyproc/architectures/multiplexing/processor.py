
import torch
import torch.nn as nn
import numpy as np
from bspyproc.processors.processor_mgr import get_processor
from bspyproc.utils.pytorch import TorchUtils


class ArchitectureProcessor():
    def __init__(self, configs):
        self.configs = configs
        self.processor = get_processor(configs)
        self.clipping_value = configs['waveform']['output_clipping_value'] * self.get_amplification_value()
        self.conversion_offset = -0.6

    def clip(self, x):
        x[x > self.clipping_value] = self.clipping_value
        return x

    def batch_norm(self, bn, x1, x2):
        x1 = TorchUtils.get_tensor_from_numpy(x1)
        x2 = TorchUtils.get_tensor_from_numpy(x2)
        h = bn(torch.cat((x1, x2), dim=1))
        std1 = np.sqrt(torch.mean(bn.running_var).cpu().numpy())
        cut = 2 * std1
        # Pass it through output layer
        h = torch.tensor(1.8 / (4 * std1)) * \
            torch.clamp(h, min=-cut, max=cut) + self.conversion_offset
        return h.numpy()

    def get_amplification_value(self):
        return self.processor.get_amplification_value()


class TwoToOneProcessor(ArchitectureProcessor):
    def __init__(self, configs):
        super().__init__(configs)
        if configs['batch_norm']:
            self.bn1 = nn.BatchNorm1d(2, affine=False)
            self.process_layer1 = self.process_layer1_batch_norm
        else:
            self.process_layer1 = self.process_layer1_alone

    def get_output(self, x):
        x1 = self.processor.get_output(x[:, 0:7])
        x2 = self.processor.get_output(x[:, 7:14])

        x = self.process_layer1(x, x1, x2)
        result = self.processor.get_output(x[:, 14:])

        return self.process_output_layer(result)

    def process_layer1_alone(self, x, x1, x2):
        x[:, 14] = self.clip(x1[:, 0])
        x[:, 15] = self.clip(x2[:, 0])
        return x

    def process_layer1_batch_norm(self, x, x1, x2):
        bnx = self.batch_norm(self.bn1, x1, x2)

        x[:, 14] = self.clip(bnx[:, 0])
        x[:, 15] = self.clip(bnx[:, 1])
        return x

    def process_output_layer(self, y):
        return self.clip(y)


class TwoToTwoToOneProcessor(ArchitectureProcessor):
    def __init__(self, configs):
        super().__init__(configs)
        if configs['batch_norm']:
            self.bn1 = nn.BatchNorm1d(2, affine=False)
            self.bn2 = nn.BatchNorm1d(2, affine=False)
            self.process_layer1 = self.process_layer1_batch_norm
            self.process_layer2 = self.process_layer2_batch_norm
        else:
            self.process_layer1 = self.process_layer1_alone
            self.process_layer2 = self.process_layer2_alone

    def get_output(self, x):

        x1 = self.processor.get_output(x[:, 0:7])
        x2 = self.processor.get_output(x[:, 7:14])

        x = self.process_layer1(x, x1, x2)

        h1 = self.processor.get_output(x[:, 14:21])
        h2 = self.processor.get_output(x[:, 21:28])

        x = self.process_layer2(x, h1, h2)

        result = self.processor.get_output(x[:, 28:])

        return self.process_output_layer(result)

    def process_layer1_alone(self, x, x1, x2):
        x[:, 14] = self.clip(x1[:, 0])
        x[:, 15] = self.clip(x2[:, 0])
        x[:, 21] = self.clip(x1[:, 0])
        x[:, 22] = self.clip(x2[:, 0])
        return x

    def process_layer1_batch_norm(self, x, x1, x2):
        bnx = self.batch_norm(self.bn1, x1, x2)

        x[:, 14] = self.clip(bnx[:, 0])
        x[:, 15] = self.clip(bnx[:, 1])
        x[:, 21] = self.clip(bnx[:, 0])
        x[:, 22] = self.clip(bnx[:, 1])
        return x

    def process_layer2_alone(self, x, x1, x2):
        x[:, 28] = self.clip(x1[:, 0])
        x[:, 29] = self.clip(x2[:, 0])
        return x

    def process_layer2_batch_norm(self, x, x1, x2):
        bnx = self.batch_norm(self.bn2, x1, x2)

        x[:, 28] = self.clip(bnx[:, 0])
        x[:, 29] = self.clip(bnx[:, 0])
        return x

    def process_output_layer(self, y):
        return self.clip(y)
