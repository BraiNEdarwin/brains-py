"""

"""
import sys
import math
import signal
import threading

import numpy as np
import nidaqmx.system.device as device

from threading import Thread

from brainspy.processors.hardware.drivers.ni.tasks import get_tasks_driver
from brainspy.processors.hardware.drivers.ni.channels import init_channel_names

# from brainspy.utils.control import get_control_voltage_indices, merge_inputs_and_control_voltages_in_numpy


# SECURITY FLAGS.
# WARNING - INCORRECT VALUES FOR THESE FLAGS CAN RESULT IN DAMAGING THE DEVICES
INPUT_VOLTAGE_THRESHOLD = 1.5
CDAQ_TO_NIDAQ_RAMPING_TIME_SECONDS = 0.1
CDAQ_TO_CDAQ_RAMPING_TIME_SECONDS = 0.03
SYNCHRONISATION_VALUE = 0.04  # do not reduce to less than 0.02


class NationalInstrumentsSetup():

    def __init__(self, configs):
        self.init_configs(configs)
        self.init_tasks(configs['driver'])
        self.enable_os_signals()
        self.init_semaphore()

    def init_configs(self, configs):
        self.configs = configs
        self.last_shape = -1
        self.data_results = None
        self.offsetted_shape = None
        self.ceil = None

        if configs["max_ramping_time_seconds"] == 0:
            input(
                "WARNING: IF YOU PROCEED THE DEVICE CAN BE DAMAGED. READ THIS MESSAGE CAREFULLY. \n The security check for the ramping time has been disabled. Steep rampings can can damage the device. Proceed only if you are sure that you will not damage the device. If you want to avoid damagesimply exit the execution. \n ONLY If you are sure about what you are doing press ENTER to continue. Otherwise STOP the execution of this program."
            )
        assert (
            configs["data"]["waveform"]["slope_length"] / configs["driver"]["sampling_frequency"]
            >= configs["max_ramping_time_seconds"]
        )

    def init_tasks(self, configs):
        self.tasks_driver = get_tasks_driver(configs)

        activation_channel_names, readout_channel_names, self.instruments = init_channel_names(configs)

        # TODO: add a maximum and a minimum to the activation channels
        self.tasks_driver.init_activation_channels(activation_channel_names)
        self.tasks_driver.init_readout_channels(readout_channel_names)

    def init_semaphore(self):
        global event
        global semaphore
        event = threading.Event()
        semaphore = threading.Semaphore()

    def reset(self):
        self.close_tasks()
        for instrument in self.instruments:
            device.Device(name=instrument).reset_device()

    def process_output_data(self, data):
        return np.array([data]) * self.configs["driver"]["amplification"]  # Creates a numpy array from a list with dimensions (n,1) and multiplies it by the amplification of the device

    def read_data(self, y):
        global p

        p = Thread(target=self._read_data, args=(y,))
        if not event.is_set():
            semaphore.acquire()
            p = Thread(target=self._read_data, args=(y,))
            p.start()
            p.join()
            if self.data_results is None:
                print("Nothing could be read. Stopping program")
                self.os_signal_handler(None)
            semaphore.release()
        return self.data_results

    def set_shape_vars(self, shape):
        if self.last_shape != shape:
            self.last_shape = shape
            self.tasks_driver.set_shape(self.configs["driver"]["sampling_frequency"], shape)
            self.offsetted_shape = shape + self.configs["offset"]
            self.ceil = (
                math.ceil((self.offsetted_shape) / self.configs["driver"]["sampling_frequency"]) + 1
            )

    def is_hardware(self):
        return True

    def _read_data(self, y):
        """
        y = It represents the input data as matrix where the shpe is defined by the "number of inputs to the device" times "input points that you want to input to the device".
        """
        self.data_results = None
        self.read_security_checks(y)
        self.set_shape_vars(y.shape[1])

        self.tasks_driver.start_tasks(y, self.configs["auto_start"])
        read_data = self.tasks_driver.read(self.offsetted_shape, self.ceil)
        self.tasks_driver.stop_tasks()

        self.data_results = read_data
        return read_data

    def read_security_checks(self, y):
        for n, y_i in enumerate(y):
            assert all(
                y_i < INPUT_VOLTAGE_THRESHOLD
            ), f"Voltages in electrode {n} higher ({y_i.max()}) than the max. allowed value ({INPUT_VOLTAGE_THRESHOLD} V)"
            assert all(
                y_i > -INPUT_VOLTAGE_THRESHOLD
            ), f"Voltages in electrode {n} lower ({y_i.min()}) than the min. allowed value ({-INPUT_VOLTAGE_THRESHOLD} V)"
            assert (
                y_i[0] == 0.0
            ), f"First value of input stream in electrode {n} is non-zero ({y_i[0]})"
            assert (
                y_i[-1] == 0.0
            ), f"Last value of input stream in electrode {n} is non-zero ({y_i[-1]})"

    def close_tasks(self):
        self.tasks_driver.close_tasks()

    def get_amplification_value(self):
        return self.configs["driver"]["amplification"]

    def forward_numpy(self):
        pass

    # These functions are used to handle the termination of the read task in such a way that enables the last read to finish, and closes the tasks afterwards

    def os_signal_handler(self, signum, frame=None):
        event.set()
        print(
            "Interruption/Termination signal received. Waiting for the reader to finish."
        )
        p.join()
        print("Closing nidaqmx tasks")
        self.close_tasks()
        sys.exit(0)

    def enable_os_signals(self):
        if sys.platform == "win32":
            import win32api
            win32api.SetConsoleCtrlHandler(self.os_signal_handler, True)
        else:
            signal.signal(signal.SIGTERM, self.os_signal_handler)
            signal.signal(signal.SIGINT, self.os_signal_handler)

    def disable_os_signals(self):
        if sys.platform == "win32":
            import win32api  # ignoring the signal
            win32api.SetConsoleCtrlHandler(None, True)
        else:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            signal.signal(signal.SIGINT, signal.SIG_IGN)
