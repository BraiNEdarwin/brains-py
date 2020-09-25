import numpy as np


def init_channel_data(configs):
    if configs['instruments_setup']['device_no'] == "single":
        instruments = []
        activation_channel_list = init_activation_channels(configs['instruments_setup'], activation_channel_list=[])
        readout_channel_list = init_readout_channels(configs['instruments_setup'], readout_channel_list=[])
        instruments = add_uniquely(instruments, configs['instruments_setup']['activation_instrument'])
        instruments = add_uniquely(instruments, configs['instruments_setup']['readout_instrument'])
        voltage_ranges = init_voltage_ranges(configs['instruments_setup']['min_activation_voltages'], configs['instruments_setup']['max_activation_voltages'])
    elif configs['instruments_setup']['device_no'] == "multiple":
        instruments = []
        activation_channel_list = []
        readout_channel_list = []
        voltage_ranges_list = []
        for device_name in configs['instruments_setup']:
            if device_name != 'trigger_source' and device_name != 'device_no':
                mask = get_mask(configs['instruments_setup'][device_name])
                if mask is None or sum(mask > 0):
                    configs['instruments_setup'][device_name]['activation_channels'] = list(np.array(configs['instruments_setup'][device_name]['activation_channels'])[mask == 1])
                    activation_channel_list = init_activation_channels(configs['instruments_setup'][device_name], activation_channel_list=activation_channel_list)
                    readout_channel_list = init_readout_channels(configs['instruments_setup'][device_name], readout_channel_list=readout_channel_list)
                    instruments = add_uniquely(instruments, configs['instruments_setup'][device_name]['activation_instrument'])
                    instruments = add_uniquely(instruments, configs['instruments_setup'][device_name]['readout_instrument'])
                    voltage_ranges = init_voltage_ranges(configs['instruments_setup'][device_name]['min_activation_voltages'], configs['instruments_setup'][device_name]['max_activation_voltages'])
                    if mask is not None:
                        voltage_ranges = voltage_ranges[mask == 1]
                    voltage_ranges_list.append(voltage_ranges)
        voltage_ranges = concatenate_voltage_ranges(voltage_ranges_list)

    else:
        print('Error in driver configuration devices device_no, select either single or multiple.')
    return activation_channel_list, readout_channel_list, instruments, voltage_ranges


def init_voltage_ranges(min_voltages, max_voltages, mask=None):
    min_voltages = np.array(min_voltages)[:, np.newaxis]
    max_voltages = np.array(max_voltages)[:, np.newaxis]
    if mask is not None:
        min_voltages = min_voltages[mask == 1]
        max_voltages = max_voltages[mask == 1]
    voltage_ranges = np.concatenate((min_voltages, max_voltages), axis=1)
    return voltage_ranges


def concatenate_voltage_ranges(voltage_ranges):
    result = voltage_ranges[0]
    for i in range(1, len(voltage_ranges)):
        result = np.concatenate((result, voltage_ranges[i]), axis=0)
    return result


def init_activation_channels(configs, activation_channel_list=[]):
    for i in range(len(configs['activation_channels'])):
        activation_channel_list.append(configs['activation_instrument'] + "/ao" + str(configs['activation_channels'][i]))
    return activation_channel_list


def init_readout_channels(configs, readout_channel_list=[]):
    for i in range(len(configs['readout_channels'])):
        readout_channel_list.append(configs['readout_instrument'] + "/ai" + str(configs['readout_channels'][i]))

    return readout_channel_list


def get_mask(configs):
    if 'activation_channel_mask' in configs:
        return np.array(configs['activation_channel_mask'])
    else:
        return None


def add_uniquely(original_list, value):
    if value not in original_list:
        original_list.append(value)
    return original_list


if __name__ == "__main__":
    from brainspy.utils.io import load_configs

    configs = load_configs('/home/unai/Documents/3-programming/brainspy-tasks/configs/defaults/processors/hw.yaml')

    a, r, ins, vr = init_channel_data(configs['driver'])
    print(a)
    print(r)
