"""

"""
import sys
import math
import warnings
import signal
import threading
import numpy as np
from threading import Thread
from brainspy.processors.hardware.drivers.ni.tasks import get_tasks_driver

"""
SECURITY FLAGS.
WARNING - INCORRECT VALUES FOR THESE FLAGS CAN RESULT IN DAMAGING THE DEVICES
"""
INPUT_VOLTAGE_THRESHOLD = 1.5
CDAQ_TO_NIDAQ_RAMPING_TIME_SECONDS = 0.1
CDAQ_TO_CDAQ_RAMPING_TIME_SECONDS = 0.03
SYNCHRONISATION_VALUE = 0.04  # do not reduce to less than 0.02


class NationalInstrumentsSetup:
    def __init__(self, configs):
        """
        This method invokes 4 other methods to :
            1. initialize the configurations of the setup

            2. initialize the semaphore which will manage the main thread by synchronsing it with the read/write of data

            3. Enable OS signals to support read/write in both linux and windows based operating systems at all times.
               This ensures security for the device which can be interrupted with a Ctrl+C command if something goes wrong.
               These functions are used to handle the termination of the read task in such a way that enables the last read to finish,
               and closes the tasks afterwards

            4. To intialize the tasks driver based on the configurations

        Parameters
        ----------
        configs : dict
            configurations of the model as a python dictionary
        """
        self.init_configs(configs)
        self.init_tasks(configs)
        self.enable_os_signals()
        self.init_semaphore()

    def init_configs(self, configs):
        """
        To initialise the configurations of the setup.
        Note - The configurations dictionary contains a "max_ramping_time_seconds" key whose value should be chosen carefully because steep values may damage the device.

        Parameters
        ----------
        configs : dict
            configurations of the model as a python dictionary

        Data key,value pairs required in the configs to initialise the setup classes

            max_ramping_time_seconds : int - To set the ramp time for the setup
                                            WARNING -The security check for the ramping time has been disabled. Steep rampings can can damage the device.
            offset : int - To set the offset value of the wave
            auto_start : bool - Too auto start the setup tasks or not
            data:
                waveform:
                    plateau_length: int - A plateau of at least 3 is needed to train the perceptron (That requires at least 10 values (3x4 = 12)).
                    slope_length : int - Length of the slope of a waveform
            driver:
                sampling_frequency: int - defines the number of samples per second (or per other unit) taken from a continuous signal
                amplification: float - To set the amplification value of the voltages
        """
        self.configs = configs
        self.last_shape = -1
        self.data_results = None
        self.offsetted_shape = None
        self.ceil = None

        print(f"Sampling frequency: {configs['sampling_frequency']}")
        print(f"Max ramping time: {configs['max_ramping_time_seconds']} seconds. ")
        if configs["max_ramping_time_seconds"] == 0:
            input(
                "WARNING: IF YOU PROCEED THE DEVICE CAN BE DAMAGED. READ THIS MESSAGE CAREFULLY. \n The security check for the ramping time has been disabled. Steep rampings can can damage the device. Proceed only if you are sure that you will not damage the device. If you want to avoid damage simply exit the execution. \n ONLY If you are sure about what you are doing press ENTER to continue. Otherwise STOP the execution of this program."
            )

    def init_tasks(self, configs):
        """
        To intialize the tasks driver and voltage ranges based on the configurations.

        Parameters
        ----------
        configs : dict
            configurations of the model as a python dictionary
        """
        self.tasks_driver = get_tasks_driver(configs)
        self.tasks_driver.init_tasks(configs)
        self.voltage_ranges = (
            self.tasks_driver.voltage_ranges
        )  # To be improved, it should have the same form to be accessed by both SurrogateModel (SoftwareProcessor) and driver.

    def init_semaphore(self):
        """
        To initialize the semaphore which will manage the main thread by synchronsing it with the read/write of data
        """
        global event
        global semaphore
        event = threading.Event()
        semaphore = threading.Semaphore()

    def reset(self):
        """
        To reset the tasks driver by closing all tasks
        """
        self.tasks_driver.close_tasks()

    def process_output_data(self, data):
        """
        To processs the output data.
        The function creates a numpy array from a list with dimensions (n,1) and multiplies it by the amplification of the device.
        It is transposed to enable the multiplication of multiple outputs by an array of amplification values.

        Parameters
        ----------
        data : list
            output data

        Returns
        -------
        np.array
            processed output data computed from the amplification value
        """
        data = np.array(data)
        if len(data.shape) == 1:
            data = data[np.newaxis, :]
        return (data.T * self.configs["amplification"]).T

    def read_data(self, y):
        """
        initializes the semaphore to read data from the device
        if the data cannot be read, a signal is sent to the signal handler which blocks the calling thread and closes the nidaqmx tasks

        Parameters
        ----------
        y : np.array
            It represents the input data as matrix where the shpe is defined by
            the "number of inputs to the device" times "input points that you want to input to the device".

        Returns
        -------
        list
            data read from the device
        """
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

    def set_shape_vars(self, shape):  # TODO
        """

        Parameters
        ----------
        shape : [type]
            [description]
        """
        if self.last_shape != shape:
            self.last_shape = shape
            self.tasks_driver.set_shape(
                self.configs["sampling_frequency"], shape
            )
            self.offsetted_shape = shape + self.configs["offset"]
            self.ceil = (
                math.ceil(
                    (self.offsetted_shape)
                    / self.configs["sampling_frequency"]
                )
                + 1
            )

    def is_hardware(self):
        """
        To check if the device is a hardware or not

        Returns
        -------
        bool
            True or False based on wheather the device is a hardware or not
        """
        return True

    def _read_data(self, y):
        """
        To read data from the device

        Parameters
        -----------
        y : np.array
            It represents the input data as matrix where the shape is defined by the "number of inputs to the device" times "input points that you want to input to the device".

        Returns
        --------
        np.array
            Data read from the device

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
        """
        This method reads the security checks from the input data, and makes sure that the input voltage does not go above certain threshhold

        Parameters
        ----------
         y : np.array
            It represents the input data as matrix where the shape is defined by the "number of inputs to the device" times "input points that you want to input to the device".
        """
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
        """
        To close all tasks currently running on this device
        """
        self.tasks_driver.close_tasks()

    def get_amplification_value(self):
        """
        To get the amplification value from the data provided in the configuratons dictionary

        Returns
        -------
        int
            amplification value
        """
        return self.configs["driver"]["amplification"]

    def forward_numpy(self):
        """
        The forward function computes output numpy values from input numpy array.
        This is done to enable compatibility of the the model which is an nn.Module with numpy
        """
        pass

    def os_signal_handler(self, signum, frame=None):
        """
        used to handle the termination of the read task in such a way that enables the last read to finish, and closes the tasks afterwards

        Parameters
        ----------
        signum : [type]
            [description]
        frame : [type], optional
            [description], by default None
        """
        event.set()
        print(
            "Interruption/Termination signal received. Waiting for the reader to finish."
        )
        p.join()
        print("Closing nidaqmx tasks")
        self.close_tasks()
        sys.exit(0)

    def enable_os_signals(self):
        """
        To enable the OS signals by adding an a signal HandlerRoutine to support read/write in both linux and windows based operating systems
        """
        import win32api

        if sys.platform == "win32":
            win32api.SetConsoleCtrlHandler(self.os_signal_handler, True)
        else:
            signal.signal(signal.SIGTERM, self.os_signal_handler)
            signal.signal(signal.SIGINT, self.os_signal_handler)

    def disable_os_signals(self):
        """
        To disable the OS signals by removing the signal HandlerRoutine in the the win32 OS or ignoring the signal incase of other processors
        """
        import win32api

        if sys.platform == "win32":

            win32api.SetConsoleCtrlHandler(None, True)
        else:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            signal.signal(signal.SIGINT, signal.SIG_IGN)
