import torch
from brainspy.processors.dnpu import DNPU
from brainspy.processors.processor import Processor
from brainspy.utils.pytorch import TorchUtils


def get_configs():
    """
    Generate the sample configs for the Task Manager
    """
    configs = {}
    configs["waveform"] = {}
    configs["waveform"]["plateau_length"] = 10
    configs["waveform"]["slope_length"] = 30

    configs["amplification"] = 100
    configs["inverted_output"] = True
   #configs["output_clipping_range"] = [-1, 1]

    configs["instrument_type"] = "cdaq_to_cdaq"
    configs["instruments_setup"] = {}

    # TODO Specify Instrument type
    # For a CDAQ setup, cdaq_to_cdaq.
    # For a NIDAQ setup, cdaq_to_nidaq.
    configs["processor_type"] = configs["instrument_type"] = "cdaq_to_cdaq"

    # TODO Specify the name of the Trigger Source
    configs["instruments_setup"]["trigger_source"] = "cDAQ2/segment1"

    # TODO Specify the name of the Activation instrument
    configs["instruments_setup"]["activation_instrument"] = "cDAQ2Mod3"

    # TODO Specify the Activation channels (pin numbers)
    # For example, [1,2,3,4,5,6,7]
    configs["instruments_setup"]["activation_channels"] = [0, 1, 2, 3, 4, 5, 6]
    configs["instruments_setup"]["activation_channel_mask"] = [1, 1, 1, 1, 1, 1, 1]
    # TODO Specify the activation Voltage ranges
    # For example, [[-1.2, 0.6],[-1.2, 0.6],[-1.2, 0.6],[-1.2, 0.6],[-1.2, 0.6],[-0.7, 0.3],[-0.7, 0.3]]
    configs["instruments_setup"]["activation_voltage_ranges"] = [[-1.2, 0.6],
                                                                 [-1.2, 0.6],
                                                                 [-1.2, 0.6],
                                                                 [-1.2, 0.6],
                                                                 [-1.2, 0.6],
                                                                 [-1.2, 0.6],
                                                                 [-1.2, 0.6]]

    # TODO Specify the name of the Readout Instrument
    configs["instruments_setup"]["readout_instrument"] = "cDAQ2Mod1"

    # TODO Specify the readout channels
    # For example, [4]
    configs["instruments_setup"]["readout_channels"] = [0]

    configs["instruments_setup"]["activation_sampling_frequency"] = 5000
    
    configs["instruments_setup"]["readout_sampling_frequency"] = 10000
    configs["instruments_setup"]["average_io_point_difference"] = True
    configs["instruments_setup"]["multiple_devices"] = False

    configs['max_ramping_time_seconds'] = 1
    configs["inverted_output"] = True
    configs["amplification"] = [100]
    configs["plateau_length"] = 5
    configs["slope_length"] = 10
    configs["offset"] = 0
    configs['auto_start'] = True
    return configs


def check_test_configs(test_dict):
    """
    Check if a value is present in a dict

    This is a helper function to test files which require connection
    to the hardware.
    """
    check = False
    for key, val in test_dict.items():
        if check:
            break
        if type(val) == str:
            if not val:
                check = True
                break
        if type(val) == list:
            if len(val) == 0:
                check = True
                break
        if type(val) == dict:
            check = check_test_configs(val)
    return check


def get_custom_model_configs():
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
    return configs, model_data


def get_random_model_state_dict():
    return torch.load('tests/data/random_state_dict.pt',
                      map_location=TorchUtils.get_device())


class CustomLogger():
    def log_performance(self, train_losses, val_losses, epoch):
        pass

    def log_train_step(self, epoch, inputs, targets, predictions, model, loss,
                       running_loss):
        pass

    def log_val_step(self, epoch, inputs, targets, predictions, model, loss,
                     val_loss):
        pass

    def close(self):
        pass


def is_hardware_fake():
    return True


def fake_criterion(outputs, targets):
    return -TorchUtils.format(torch.ones_like(targets[0]))


class DefaultCustomModel(torch.nn.Module):
    def __init__(self, configs, info, forward_pass_type='vec'):
        super(DefaultCustomModel, self).__init__()
        self.node_no = 1
        self.gamma = 1

        processor = Processor(configs, info)
        self.dnpu = DNPU(processor=processor,
                         data_input_indices=[configs['input_indices']] *
                         self.node_no,
                         forward_pass_type=forward_pass_type)
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

    def constraint_weights(self):
        self.dnpu.constraint_control_voltages()

    def format_targets(self, x: torch.Tensor) -> torch.Tensor:
        return self.dnpu.format_targets(x)

    ################################################################
    #  If you want to implement on-chip GA, you need these functions
    def is_hardware(self):
        return self.dnpu.processor.is_hardware()

    def close(self):
        self.dnpu.close()
