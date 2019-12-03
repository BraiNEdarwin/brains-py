
from bspyproc.processors.processor_mgr import get_processor


class TwoToOneProcessor():
    def __init__(self, configs):
        self.configs = configs
        self.processor = get_processor(configs)

    def get_output(self, x):
        # Pass through input layer
        # x = (self.scale * x) + self.offset
        # configs "input_indices": [0, 1]
        x1 = self.processor.get_output(x[:, 0:7])
        x2 = self.processor.get_output(x[:, 7:14])
        # [ i1, i2, c1 , c2, c3, ...  ]
        # x [0:21]
        # x[0:7,:] = [ i1, i2, c1, c2, c3, c4, c5]
        # x[7:14] =  [i1, i2, c1, c2, c3, c4, c5]
        # x[14:21] =  [x1, x2, c1, c2, c3, c4, c5]
        # [i1, i2, c1, c2, c3, c4, c5]
        # --- BatchNorm --- #
        # h = self.bn1(torch.cat((x1, x2), dim=1))
        # std1 = np.sqrt(torch.mean(self.bn1.running_var).cpu().numpy())
        # cut = 2 * std1
        # Pass it through output layer
        # h = torch.tensor(1.8 / (4 * std1)) * \
        #    torch.clamp(h, min=-cut, max=cut) + self.conversion_offset

        x[:, 7] = x1[:, 0]
        x[:, 8] = x2[:, 0]
        result = self.processor.get_output(x[:, 14:])
        # --- BatchNorm --- #
        # h = self.bn1(torch.cat((x1, x2), dim=1))
        # std1 = np.sqrt(torch.mean(self.bn1.running_var).cpu().numpy())
        # cut = 2 * std1
        # Pass it through output layer
        # h = torch.tensor(1.8 / (4 * std1)) * \
        #    torch.clamp(h, min=-cut, max=cut) + self.conversion_offset

        return result

    def get_amplification_value(self):
        return self.processor.get_amplification_value()


class TwoToTwoToOneProcessor():
    def __init__(self, configs):
        self.configs = configs
        self.processor = get_processor(configs)

    def get_output(self, x):
        # Pass through input layer
        # x = (self.scale * x) + self.offset
        # configs "input_indices": [0, 1]
        x1 = self.processor.get_output(x[0:7])
        x2 = self.processor.get_output(x[7:14])

        # --- BatchNorm --- #
        # h = self.bn1(torch.cat((x1, x2), dim=1))
        # std1 = np.sqrt(torch.mean(self.bn1.running_var).cpu().numpy())
        # cut = 2 * std1
        # Pass it through output layer
        # h = torch.tensor(1.8 / (4 * std1)) * \
        #    torch.clamp(h, min=-cut, max=cut) + self.conversion_offset

        x[14 + self.configs['input_indices'][0]] = x1
        x[14 + self.configs['input_indices'][1]] = x2
        x[21 + self.configs['input_indices'][0]] = x1
        x[21 + self.configs['input_indices'][1]] = x2

        h1 = self.processor.get_output(x[14:21])
        h2 = self.processor.get_output(x[21:28])
        # --- BatchNorm --- #
        # h = self.bn1(torch.cat((x1, x2), dim=1))
        # std1 = np.sqrt(torch.mean(self.bn1.running_var).cpu().numpy())
        # cut = 2 * std1
        # Pass it through output layer
        # h = torch.tensor(1.8 / (4 * std1)) * \
        #    torch.clamp(h, min=-cut, max=cut) + self.conversion_offset

        x[28 + self.configs['input_indices'][0]] = h1
        x[28 + self.configs['input_indices'][1]] = h2
        result = self.processor.get_output(x[28:])

        # --- BatchNorm --- #
        # h = self.bn1(torch.cat((x1, x2), dim=1))
        # std1 = np.sqrt(torch.mean(self.bn1.running_var).cpu().numpy())
        # cut = 2 * std1
        # Pass it through output layer
        # h = torch.tensor(1.8 / (4 * std1)) * \
        #    torch.clamp(h, min=-cut, max=cut) + self.conversion_offset

        return result

    def get_amplification_value(self):
        return self.processor.get_amplification_value()
