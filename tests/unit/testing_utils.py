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
    configs["output_clipping_range"] = [-1, 1]

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
    configs["instruments_setup"]["activation_channels"] = [
        0, 1, 2, 3, 4, 5, 6
    ]
    # TODO Specify the activation Voltage ranges
    # For example, [[-1.2, 0.6],[-1.2, 0.6],[-1.2, 0.6],[-1.2, 0.6],[-1.2, 0.6],[-0.7, 0.3],[-0.7, 0.3]]
    configs["instruments_setup"]["activation_voltage_ranges"] = [
        [-1.2, 0.6], [-1.2, 0.6], [-1.2, 0.6], [-1.2, 0.6], [-1.2, 0.6],
        [-1.2, 0.6], [-1.2, 0.6]]
    

    # TODO Specify the name of the Readout Instrument
    configs["instruments_setup"]["readout_instrument"] = "cDAQ2Mod1"

    # TODO Specify the readout channels
    # For example, [4]
    configs["instruments_setup"]["readout_channels"] = [0]

    configs["instruments_setup"]["activation_sampling_frequency"] = 5000
    configs["instruments_setup"]["readout_sampling_frequency"] = 10000
    configs["instruments_setup"]["average_io_point_difference"] = True
    configs["instruments_setup"]["multiple_devices"] = False

    configs['driver'] = {}
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
