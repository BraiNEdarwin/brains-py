import torch
from brainspy.processors.dnpu import DNPU

from brainspy.processors.processor import Processor


class DefaultCustomModel(torch.nn.Module):

    def __init__(self):
        super(DefaultCustomModel, self).__init__()
        self.node_no = 1
        self.gamma = 1
        model_data = {}
        model_data["info"] = {}
        model_data["info"]["model_structure"] = {
            "hidden_sizes": [90, 90, 90],
            "D_in": 7,
            "D_out": 1,
            "activation": "relu",
        }
        model_data["info"]["electrode_info"] = {
            'electrode_no': 8,
            'activation_electrodes': {
                'electrode_no':
                7,
                'voltage_ranges': [[-1.2, 0.6], [-1.2, 0.6], [-1.2, 0.6],
                                   [-1.2, 0.6], [-1.2, 0.6], [-0.7, 0.3],
                                   [-0.7, 0.3]]
            },
            'output_electrodes': {
                'electrode_no': 1,
                'amplification': [100],
                'clipping_value': None
            }
        }
        configs = {}
        configs["processor_type"] = "simulation"
        configs['input_indices'] = [3, 4]
        configs["waveform"] = {}
        configs["waveform"]["plateau_length"] = 10
        configs["waveform"]["slope_length"] = 30
        processor = Processor(configs, model_data['info'])
        self.dnpu = DNPU(processor=processor,
                         data_input_indices=[configs['input_indices']] *
                         self.node_no,
                         forward_pass_type='vec')
        self.dnpu.add_input_transform([-1, 1])

    def forward(self, x):
        x = self.dnpu(x)
        return x

    # If you want to swap from simulation to hardware, or vice-versa you need these functions
    def hw_eval(self, configs, info=None):
        self.eval()
        self.dnpu.hw_eval(configs, info)

    def sw_train(self, configs, info=None, model_state_dict=None):
        self.train()
        self.dnpu.sw_train(configs, info, model_state_dict)

    ##########################################################################################
    # If you want to be able to get information about the ranges from outside, you have to add the
    #  following functions.
    def get_input_ranges(self):
        return self.dnpu.get_input_ranges()

    def get_control_ranges(self):
        return self.dnpu.get_control_ranges()

    def get_control_voltages(self):
        return self.dnpu.get_control_voltages()

    def set_control_voltages(self, control_voltages):
        self.dnpu.set_control_voltages(control_voltages)

    def get_clipping_value(self):
        return self.dnpu.get_clipping_value()

    # For being able to maintain control voltages within ranges, you should implement the following functions
    #  (only those which you are planning to use)
    def regularizer(self):
        return self.gamma * (self.dnpu.regularizer())

    def set_regul_factor(self, factor):
        pass

    def constraint_weights(self):
        print()

    def constraint_control_voltages(self):
        self.dnpu.constraint_control_voltages()

    def format_targets(self, x: torch.Tensor) -> torch.Tensor:
        return self.dnpu.format_targets(x)

    ################################################################
    #  If you want to implement on-chip GA, you need these functions
    def is_hardware(self):
        return self.dnpu.processor.is_hardware

    def close(self):
        self.dnpu.close()
