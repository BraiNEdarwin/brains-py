import numpy as np
from typing import Union
"""
This file contains a set of functions that are used to initialise the channels used to
specify the connection between the National Instrument devices and the electrodes of
DNPU hardware devices. It supports to declare channels for one or more DNPU hardware
devices simultaneously.
"""


def init_channel_data(configs):
    """
    It creates a single array of voltage ranges, as defined in the activation_voltage_ranges
    flag of the driver's configs files. It is meant for cases where the PCB supports writing
    to more than one DNPU hardware device simultaneously, in order to concatenate all input
    voltage ranges for all activation electrodes into a single numpy array.

    Parameters
    ----------
    configs: dict
        Dictionary containing information about the instruments setup. The dictionary contains
        the following keys:
            instruments_setup:
                multiple_devices: boolean
                    Whether if the configurations contain the configurations of reading from a
                    single DNPU hardware device or multiple DNPU hardware devices.
                trigger_source: str
                    For synchronisation purposes, sending data for the activation voltages on one NI
                    Task can trigger the readout device of another NI Task. In these cases, the
                    trigger source name should be specified in the configs. This is only applicable
                    for CDAQ to
                    CDAQ setups (with or without real-time rack).
                    E.g., cDAQ1/segment1 - More information at:
                    https://nidaqmx-python.readthedocs.io/en/latest/start_trigger.html
                The reminder of the keys work as follows. If the attribute multiple_devices is set
                to False, the following configurations apply:
                    activation_instrument: str
                        The name of the National Instument device used for writing the data into the
                        activation electrodes of the hardware DNPU device.
                        (e.g., cDAQ1Mod2)
                    activation_channels: list
                        A list containing the physical ao channels of the National Instrument device
                        that will be used for writing the data into the activation electrodes of the
                        hardware DNPU device. (e.g., [6,0,7,5,2,4,3])
                    activation_voltage_ranges: list
                        The maximum and minimum voltage values that the National Instrument device
                        will be allowed to sent through the activation electrodes of a particular
                        DNPU hardware device. The shape is (activation_electrode_no,2) where the
                        second dimension stands for minimum and maximum of the range, respectively.
                        E.g,:
                        [[-1.2,0.7],[-1.2,0.7],[-1.2,0.7],[-1.2,0.7],[-1.2,0.7],[-1.2,0.7],[-1.2,0.7]]
                    activation_channel_mask: list
                        A list of zeroes and ones, representing each of the channels that go to the
                        activationelectrodes of a particular DNPU. The list should have a length of
                        (activation_electrode_no). Each zero in the list will be a deactivated
                        channel, each one in the list will be an activated channel.
                        E.g., [0,0,0,0,0,0,0]
                    readout_instrument: str
                        Name of the instrument that is used for reading the output of a hardware
                        DNPU. E.g., cDAQ1Mod4
                    readout_channels: list
                        List of physical channels of the National Instruments device from which the
                        output of a hardware DNPU will be read. The length of the list is the same
                        as the number of readout electrodes.

                If the attribute multiple_devices is set to True, the same keys as above need to be
                encapsulated on a previous dictionary level.
                This can be repeated for more than one device, as follows. E.g., in a .yaml format,
                the configurations for two different DNPU hardware devices (A and B) can be
                declared as follows:
                    "
                    multiple_devices: True
                    trigger_source: cDAQ1/segment1
                    A:
                        activation_instrument: cDAQ1Mod2
                        activation_channels: [6,0,7,5,2,4,3] #ao
                        activation_voltage_ranges:
                        [[-1.2,0.7],[-1.2,0.7],[-1.2,0.7],[-1.2,0.7],[-1.2,0.7],[-1.2,0.7],[-1.2,0.7]]
                        activation_channel_mask: [0,0,0,0,0,0,0]
                        readout_instrument: cDAQ1Mod4
                        readout_channels: [0] # ai0
                    B:
                        activation_instrument: cDAQ1Mod1
                        activation_channels: [4,3,5,2,0,7,1] #ao
                        activation_voltage_ranges:
                        [[-1.2,0.7],[-1.2,0.7],[-1.2,0.7],[-1.2,0.7],[-1.2,0.7],[-1.2,0.7],[-1.2,0.7]]
                        activation_channel_mask: [0,0,0,0,0,0,0]
                        readout_instrument: cDAQ1Mod4
                        readout_channels: [3] #ai3
                    "
    Returns
    -------
    result: np.array
        A numpy array with the concatenation of several voltage ranges from more than one DNPU
        hardware device, in cases where the PCB supports writing to more than one DNPU hardware
        device simultaneously.
    """
    type_check(configs)
    if not configs["instruments_setup"]["multiple_devices"]:
        instruments = []
        activation_channel_list = init_activation_channels(
            configs["instruments_setup"], activation_channel_list=[])
        readout_channel_list = init_readout_channels(
            configs["instruments_setup"], readout_channel_list=[])
        instruments = add_uniquely(
            instruments, configs["instruments_setup"]["activation_instrument"])
        instruments = add_uniquely(
            instruments, configs["instruments_setup"]["readout_instrument"])
        voltage_ranges = np.array(
            configs['instruments_setup']['activation_voltage_ranges'])
    else:
        instruments = []
        activation_channel_list = []
        readout_channel_list = []
        voltage_ranges_list = []
        has_at_least_one_val = False
        for device_name in configs["instruments_setup"]:
            if is_device_name(device_name):
                mask = get_mask(configs["instruments_setup"][device_name])
                if mask is None or sum(mask) > 0:
                    has_at_least_one_val = True
                    configs["instruments_setup"][device_name][
                        "activation_channels"] = list(
                            np.array(configs["instruments_setup"][device_name]
                                     ["activation_channels"])[mask == 1])
                    activation_channel_list = init_activation_channels(
                        configs["instruments_setup"][device_name],
                        activation_channel_list=activation_channel_list,
                    )
                    readout_channel_list = init_readout_channels(
                        configs["instruments_setup"][device_name],
                        readout_channel_list=readout_channel_list,
                    )
                    instruments = add_uniquely(
                        instruments,
                        configs["instruments_setup"][device_name]
                        ["activation_instrument"],
                    )
                    instruments = add_uniquely(
                        instruments,
                        configs["instruments_setup"][device_name]
                        ["readout_instrument"],
                    )
                    voltage_ranges = np.array(
                        configs['instruments_setup'][device_name]
                        ['activation_voltage_ranges'],
                        dtype=np.double)
                    if mask is not None:
                        voltage_ranges = voltage_ranges[mask == 1]
                    voltage_ranges_list.append(voltage_ranges)
        if not has_at_least_one_val:
            raise AssertionError(
                "If multiple devices are used, at least 1 mask value of 1 device should be 1"
            )
        voltage_ranges = concatenate_voltage_ranges(voltage_ranges_list)
    return activation_channel_list, readout_channel_list, instruments, voltage_ranges


def type_check(configs):
    """
    Check the type of the configurations provided
    """
    assert type(configs["instruments_setup"]["multiple_devices"]
                ) == bool, "Multiple devices key should be of type bool"
    assert type(
        configs["instruments_setup"]
        ["trigger_source"]) == str, "trigger_source key should be of type str"

    # Assertions for a Single Device

    if not configs["instruments_setup"]["multiple_devices"]:
        assert type(configs["instruments_setup"]["activation_instrument"]
                    ) == str, "activation_instrument key should be of type str"
        assert type(configs["instruments_setup"]["readout_instrument"]
                    ) == str, "readout_instrument key should be of type str"
        assert type(configs["instruments_setup"]["activation_channels"]
                    ) == list, "activation_channels key should be of type list"
        assert type(configs["instruments_setup"]["readout_channels"]
                    ) == list, "readout_channels key should be of type list"
        assert type(
            configs["instruments_setup"]["activation_voltage_ranges"]
        ) == list or type(
            configs["instruments_setup"]["activation_voltage_ranges"]
        ) == np.ndarray, "The voltage_ranges should be of type - list or numpy array"
        assert len(
            configs["instruments_setup"]["activation_voltage_ranges"]
        ) == len(
            configs["instruments_setup"]["activation_channels"]
        ), "The length of channel_names should be equal to the length of voltage ranges"
        for voltage_range in configs["instruments_setup"][
                "activation_voltage_ranges"]:
            assert type(voltage_range) == list or type(
                voltage_range
            ) == np.ndarray, "Each voltage range should be a list of 2 values"
            assert len(
                voltage_range
            ) == 2, "Voltage range should contain 2 values : max and min"
            assert isinstance(
                voltage_range[0],
                (np.floating, float, int
                 )), "Volatge range can contain only int or float type values"
            assert isinstance(
                voltage_range[1],
                (np.floating, float, int
                 )), "Volatge range can contain only int or float type values"

    # Assertions for Multiple Devices

    else:
        for device_name in configs["instruments_setup"]:
            if is_device_name(device_name):
                assert type(
                    configs["instruments_setup"][device_name]
                    ["activation_instrument"]
                ) == str, "activation_instrument key should be of type str"
                assert type(
                    configs["instruments_setup"][device_name]
                    ["readout_instrument"]
                ) == str, "readout_instrument key should be of type str"
                assert type(
                    configs["instruments_setup"][device_name]
                    ["activation_channels"]
                ) == list, "activation_channels key should be of type list"
                assert type(
                    configs["instruments_setup"][device_name]
                    ["readout_channels"]
                ) == list, "readout_channels key should be of type list"
                assert type(
                    configs["instruments_setup"][device_name]
                    ["activation_voltage_ranges"]
                ) == list or type(
                    configs["instruments_setup"][device_name]
                    ["activation_voltage_ranges"]
                ) == np.ndarray, "The voltage_ranges should be of type - list or numpy array"
                assert len(
                    configs["instruments_setup"][device_name]
                    ["activation_voltage_ranges"]
                ) == len(
                    configs["instruments_setup"][device_name]
                    ["activation_channels"]
                ), "The length of channel_names should be equal to the length of voltage ranges"
                for voltage_range in configs["instruments_setup"][device_name][
                        "activation_voltage_ranges"]:
                    assert type(voltage_range) == list or type(
                        voltage_range
                    ) == np.ndarray, "Each voltage range should be a list of 2 values"
                    assert len(
                        voltage_range
                    ) == 2, "Voltage range should contain 2 values : max and min"
                    assert isinstance(
                        voltage_range[0], (np.floating, float, int)
                    ), "Volatge range can contain only int or float type values"
                    assert isinstance(
                        voltage_range[1], (np.floating, float, int)
                    ), "Volatge range can contain only int or float type values"


def is_device_name(key):
    """
    Checks if a configuration key from the instruments_setup configuration dictionary
    contains the name of a DNPU device, in the case of having a PCB with multiple
    DNPU devices (multiple_devices = True).

    Args:
        key (str): Configuration key to be checked against being a device name.
    Returns:
        bool: Whether if the configuration key is a device name
    """
    assert type(key) == str, "The key should be of type - str"
    return (key != "trigger_source" and key != "multiple_devices"
            and key != "activation_sampling_frequency"
            and key != "readout_sampling_frequency"
            and key != "average_io_point_difference")


def concatenate_voltage_ranges(voltage_ranges: Union[list, np.ndarray]):
    """
    It creates a single array of voltage ranges, as defined in the activation_voltage_ranges
    flag of the driver's configs files. It is meant for cases where the PCB supports writing
    to more than one DNPU hardware device simultaneously, in order to concatenate all input
    voltage ranges for all activation electrodes into a single numpy array.

    Parameters
    ----------
    voltage_ranges: np.array or list
        The maximum and minimum voltage values that the National Instrument device will be allowed
        to send through the activation electrodes of a particular DNPU hardware device.

    Returns
    -------
    result: np.array # or list
        A numpy array with the concatenation of several voltage ranges from more than one DNPU
        hardware device, in cases where the PCB supports writing to more than one DNPU hardware
        device simultaneously.
    """
    assert type(voltage_ranges) == np.ndarray or type(
        voltage_ranges
    ) == list, "Voltage ranges should be of type - list or numpy array"
    # assert voltage ranges >0
    if len(voltage_ranges) > 0:
        result = voltage_ranges[0]
        for i in range(1, len(voltage_ranges)):
            result = np.concatenate((result, voltage_ranges[i]), axis=0)
        return result


def init_activation_channels(configs, activation_channel_list=[]):
    """
    Method to retrieve all the activation channel lists from more than one
    hardware DNPU devices, in cases where the PCB supports writing to
    more than one DNPU hardware device simultaneously.
    Parameters
    ----------
    configs: dict
    Configurations of the driver. There is only one key needed to operate this function:
        activation_channels: list
            List of physical channels of the National Instruments device from which the input of a
            hardware DNPU will be read. The length of the list is the same as the number of
            activation electrodes.
        activation_instrument: str
            Name of the instrument that is used for writing the input of a hardware DNPU.

    Returns
    -------
    ractivation_channel_list: list
        A list containing all the activation channels currently registered in the list.
    """
    assert type(configs) == dict, "The configurations should be of type - dict"
    for i in range(len(configs["activation_channels"])):
        activation_channel_list.append(configs["activation_instrument"] +
                                       "/ao" +
                                       str(configs["activation_channels"][i]))
    return activation_channel_list


def init_readout_channels(configs: dict, readout_channel_list: list = []):
    """
    Method to retrieve all the readout channel lists from more than one
    hardware DNPU devices, in cases where the PCB supports reading from
    more than one DNPU hardware device simultaneously.
    Parameters
    ----------
    configs: dict
    Configurations of the driver. There is only one key needed to operate this function:
        readout_channels: list
            List of physical channels of the National Instruments device from which the output of a
            hardware DNPU will be read. The length of the list is the same as the number of readout
            electrodes.
        readout_instrument: str
            Name of the instrument that is used for reading the output of a hardware DNPU.

    Returns
    -------
    readout_channel_list: list
        A list containing all the readout channels currently registered in the list.
    """
    assert type(configs) == dict, "The configurations should be of type - dict"
    for i in range(len(configs["readout_channels"])):
        readout_channel_list.append(configs["readout_instrument"] + "/ai" +
                                    str(configs["readout_channels"][i]))

    return readout_channel_list


def get_mask(configs: dict):
    """
    This method retrieves a list of the masks for the channels from a configuration dictionary
    and transforms it into a numpy array. If the key does not exist in the dictionary returns
    None. The mask is used to deactivate some of the channels in PCBs that support reading from
    more than one DNPU hardware device simultaneously.

    Parameters
    ----------
    configs: Configurations of the driver. There is only one key needed to operate in this function:
        activation_channel_mask: Optional[list]
            A list of zeroes and ones, representing each of the channels that go to the activation
            electrodes of a particular DNPU. The list should have a length of
            (activation_electrode_no). Each zero in the list will be a deactivated channel, each one
            in the list will be an activated channel.
            E.g., for an 8 electrode hardware DNPU with 7 activation electrodes:
                activation_channel_mask = [0,1,1,1,1,1,1] will act as if the first electrode does
                not exist.

    Returns
    -------
    result: None or np.array
        Mask used to deactivate some of the channels in numpy format.

    """
    assert type(configs) == dict, "The configurations should be of type - dict"
    if "activation_channel_mask" in configs:
        return np.array(configs["activation_channel_mask"])
    else:
        return None


def add_uniquely(original_list, value):
    """
    Adds a value to a list if the value does not exist already in that list.

    Parameters
    ----------
    original_list: list
        List against which the value that wants to be inserted will be compared.
    value: any
        Value that wants to be added to the list.

    Returns
    -------
    original_list: list
        The original list containing the specified value just once.

    """
    assert type(original_list
                ) == list, "The original list param should be of type - list"
    if value not in original_list:
        original_list.append(value)
    return original_list
