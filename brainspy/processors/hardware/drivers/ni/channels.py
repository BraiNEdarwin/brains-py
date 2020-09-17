def init_channel_names(configs):
    if configs['devices']['device_no'] == "single":
        instruments = []
        activation_channel_list = init_activation_channels(configs['instruments_setup'])
        readout_channel_list = init_readout_channels(configs['instruments_setup'])
        instruments = add_uniquely(instruments, configs['activation_instrument'])
        instruments = add_uniquely(instruments, configs['readout_instrument'])

    elif configs['devices']['device_no'] == "multiple":
        instruments = []
        activation_channel_list = []
        readout_channel_list = []
        for device_name in configs['instruments_setup']:
            mask = get_mask(configs['devices'], device_name)
            masked_configs = apply_channel_masks(configs['instruments_setup'][device_name], mask=mask)
            activation_channel_list = init_activation_channels(masked_configs, activation_channel_list=activation_channel_list)
            readout_channel_list = init_readout_channels(masked_configs, readout_channel_list=readout_channel_list)
            if mask is None or sum(mask) > 0:
                instruments = add_uniquely(instruments, configs['activation_instrument'])
                instruments = add_uniquely(instruments, configs['readout_instrument'])

    else:
        print('Error in driver configuration devices device_no, select either single or multiple.')
    return activation_channel_list, readout_channel_list, instruments


def apply_channel_masks(configs, mask=None):
    result = configs.copy()
    if mask is not None:
        for j in range(len(mask)):
            if mask[j] == 0:
                del result['activation_channels'][j]

    return result


def init_activation_channels(configs, activation_channel_list=[]):
    for i in range(len(configs['activation_channels'])):
        activation_channel_list.append(configs['activation_instrument'] + "/ao" + str(configs['activation_channels'][i]))
    return activation_channel_list


def init_readout_channels(configs, readout_channel_list=[]):
    for i in range(len(configs['readout_channels'])):
        readout_channel_list.append(configs['readout_instrument'] + "/ai" + str(configs['readout_channels'][i]))

    return readout_channel_list


def get_mask(configs, device_name):
    if 'activation_channel_mask' in configs:
        return configs['activation_channel_mask'][device_name]
    else:
        return None


def add_uniquely(original_list, value):
    if value not in original_list:
        original_list.append(value)
    return original_list


if __name__ == "__main__":
    from brainspy.utils.io import load_configs

    configs = load_configs('/home/unai/Documents/3-programming/brainspy-tasks/configs/defaults/processors/hw.yaml')

    a, r = init_channel_names(configs['driver'])
    print(a)
    print(r)
