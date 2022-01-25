"""

"""
import sys
import math
import signal
import threading
import warnings
import numpy as np
from threading import Thread
from brainspy.processors.hardware.drivers.ni.tasks import IOTasksManager
"""
SECURITY FLAGS.
WARNING - INCORRECT VALUES FOR THESE FLAGS CAN RESULT IN DAMAGING THE DEVICES

General flags:

    * INPUT_VOLTAGE_THRESHOLD: The maximum voltage threshold that will be allowed to be sent to
                               devices.

Flags related to the CDAQ TO NIDAQ Setup:

    * CDAQ_TO_NIDAQ_RAMPING_TIME_SECONDS: The value will be added to the flag
                                          max_ramping_time_seconds : int
                                          This is an internal flag used to control that devices do
                                          not exceed this time threshold, as steep rampings can can
                                          damage DNPU devices. It will only apply to the CDAQ to
                                          NIDAQ setup.
    * SYNCHRONISATION_VALUE: It determines the time that will be taken to do the synchronisation
                             signal from the cdaq to the nidaq device. Do not reduce it to less
                             than 0.02

Flags related to the CDAQ TO CDAQ Setup:

    * CDAQ_TO_CDAQ_RAMPING_TIME_SECONDS: The value will be added to the flag
                                         max_ramping_time_seconds : int
                                         This is an internal flag used to control that devices do
                                         not exceed this time threshold, as steep rampings can can
                                         damage DNPU devices. It will apply to any CDAQ to CDAQ
                                         setups, with or without real-time rack.

"""

INPUT_VOLTAGE_THRESHOLD = 1.5
CDAQ_TO_NIDAQ_RAMPING_TIME_SECONDS = 0.001
CDAQ_TO_CDAQ_RAMPING_TIME_SECONDS = 0.001
SYNCHRONISATION_VALUE = 0.04  # do not reduce to less than 0.02


class NationalInstrumentsSetup:
    def __init__(self, configs):
        """
        This method invokes 4 other methods to :
            1. Initialise the configurations of the setup.

            2. Initialise the semaphore which will manage the main thread by synchronsing it with
               the read/write of data.

            3. Enable OS signals to support read/write in both linux and windows based operating
               systems at all times. This ensures security for the device which can be interrupted
               with a Ctrl+C command if something goes wrong. These functions are used to handle
               the termination of the read task in such a way that enables the last read to finish,
               and closes the tasks afterwards.

            4. To intialize the tasks driver based on the configurations.

        Parameters
        ----------
        configs : dict
            Key-value pairs required in the configs dictionary to initialise the driver are as
            follows:
                output_clipping_range: [float,float]
                    The the setups have a limit in the range they can read. They typically clip at
                    approximately +-4 V. Note that in order to calculate the clipping_range, it
                    needs to be multiplied by the amplification value of the setup. (e.g., in the
                    Brains setup the amplification is 28.5, is the clipping_value is +-4 (V),
                    therefore, the clipping value should be +-4 * 28.5, which is [-110,110] (nA) ).
                    The original clipping value of the surrogate models is obtained when running
                    the preprocessing of the data in
                    bspysmg.measurement.processing.postprocessing.post_process.

                amplification: float
                    The output current (nA) of the device is converted by the readout hardware to
                    voltage (V), because it is easier to do the readout of the device in voltages.
                    This output signal in nA is amplified by the hardware when doing this current to
                    voltage conversion, as larger signals are easier to detect. In order to obtain
                    the real current (nA) output of the device, the conversion is automatically
                    corrected in software by multiplying by the amplification value again.
                    The amplification value depends on the feedback resistance of each of the
                    setups. You can find a guide of the amplification value needed for each setup
                    at the brains-py wiki:
                    https://github.com/BraiNEdarwin/brains-py/wiki/F.-Hardware-setups-at-BRAINS-research-group

                instruments_setup:
                    multiple_devices: boolean
                        False will initialise the drivers to read from a single hardware DNPU.
                        True, will enable to read from more than one DNPU device at the same time.
                    activation_instrument: str
                        Name of the activation instrument as observed in the NI Max software.
                        E.g., cDAQ1Mod3
                    activation_sampling_frequency: int
                        The number of samples to be obtained in one second,
                        when transforming the activation signal from digital to analogue.
                    activation_channels: list
                        Channels through which voltages will be sent for activating the device
                        (both data inputs and control voltage electrodes). The channels can be
                        checked in the schematic of the DNPU device.
                        E.g., [8,10,13,11,7,12,14]
                    activation_voltage_ranges: list
                        Minimum and maximum voltage for the activation electrodes.
                        E.g., [[-1.2, 0.6], [-1.2, 0.6], [-1.2, 0.6], [-1.2, 0.6], [-1.2, 0.6],
                        [-0.7, 0.3], [-0.7, 0.3]]
                    readout_instrument: str
                        Name of the readout instrument as observed in the NI Max
                        software. E.g., cDAQ1Mod4
                    readout_sampling_frequency: int
                        The number of samples to be obtained in one second,
                        when transforming the readout signal from analogue to digital.
                    readout_channels: [2] list
                        Channels for reading the output current values. The channels can be checked
                        in the schematic of the DNPU device.
                    trigger_source: str
                        For synchronisation purposes, sending data for the activation voltages on
                        one NI Task can trigger the readout device of another NI Task. In these
                        cases, the trigger source name should be specified in the configs. This is
                        only applicable for CDAQ to CDAQ setups (with or without real-time rack).
                        E.g., cDAQ1/segment1 - More information at:
                        https://nidaqmx-python.readthedocs.io/en/latest/start_trigger.html
        """
        self.init_configs(configs)
        self.init_tasks(configs)
        self.enable_os_signals()
        self.init_semaphore()

    def init_configs(self, configs):
        """
        To initialise the configurations of the setup.
        Note - The configurations dictionary contains a "max_ramping_time_seconds" key whose value
        should be chosen carefully because steep values may damage the device.

        Parameters
        ----------
        configs : dict
            Configurations to be initialised. Described in the __init__ method of this class.

        """
        self.configs = configs
        self.last_points_to_write_val = -1
        self.data_results = None
        self.offsetted_points_to_write = None
        self.timeout = None
        self.init_sampling_configs(configs)
        if configs["inverted_output"]:
            self.inversion = -1
        else:
            self.inversion = 1
        print(
            f"DAC sampling frequency: {configs['instruments_setup']['activation_sampling_frequency']}"
        )
        print(
            f"ADC sampling frequency: {configs['instruments_setup']['readout_sampling_frequency']}"
        )
        print(f"DAC/ADC point difference: {self.io_point_difference}")
        print(
            f"Max ramping time: {configs['max_ramping_time_seconds']} seconds. "
        )
        if configs["max_ramping_time_seconds"] == 0:
            input(
                "WARNING: IF YOU PROCEED THE DEVICE CAN BE DAMAGED. READ THIS MESSAGE CAREFULLY. \n"
                +
                "The security check for the ramping time has been disabled. Steep rampings can"
                +
                " damage the device. Proceed only if you are sure that you will not damage the "
                +
                "device. If you want to avoid damage simply exit the execution. \n ONLY If you are "
                +
                "sure about what you are doing press ENTER to continue. Otherwise STOP the "
                + "execution of this program.")

    def init_sampling_configs(self, configs):
        """ Initialises configuration related to sampling.
            It saves the variable io_point_difference, which is calculated dividing the
            readout_sampling_frequency by the activation_sampling_frequency.
            It asserts that the remainder of this division between frequencies is zero.
            It raises a warning related to resolution loss if the activation_sampling_frequency
            is higher than half of the readout_sampling_frequency.

        Args:
            configs (dict):
                A dictionary containing at least the following keys:
                - instruments_setup:
                    readout_sampling_frequency: Frequency at which the ADC will sample.
                    activation_sampling_frequency: Frequency at which the DAC will sample.
        """
        self.io_point_difference = int(
            configs['instruments_setup']['readout_sampling_frequency'] /
            configs['instruments_setup']['activation_sampling_frequency'])
        assert configs['instruments_setup']['readout_sampling_frequency'] % configs[
            'instruments_setup']['activation_sampling_frequency'] == 0, (
                "Remainder of the division between readout (" +
                f"{configs['instruments_setup']['readout_sampling_frequency']} Hz) "
                +
                f" and activation ({configs['instruments_setup']['activation_sampling_frequency']} Hz)"
                + " frequencies is not zero.")
        if configs['instruments_setup']['activation_sampling_frequency'] > (
                configs['instruments_setup']['readout_sampling_frequency'] /
                2):
            warnings.warn(
                "Activation sampling frequency (" +
                f"{configs['instruments_setup']['activation_sampling_frequency']} Hz) "
                + " is higher than half of the readout frequency (" +
                f"{configs['instruments_setup']['readout_sampling_frequency']} Hz). "
                "By setting this configuration, you are losing resolution. ")

    def init_tasks(self, configs):
        """
        Initializes the tasks driver and voltage ranges based on the configurations.

        Parameters
        ----------
        configs : dict
            configurations of the model as a python dictionary
        """
        self.tasks_driver = IOTasksManager(configs)
        self.tasks_driver.init_tasks(configs)
        self.voltage_ranges = (
            self.tasks_driver.voltage_ranges
        )  # To be improved, it should have the same form to be accessed by both
        # SurrogateModel (SoftwareProcessor) and driver.

    def init_semaphore(self):
        """
        Initializes the semaphore that will manage the main thread by synchronsing it with the
        read/write of data.
        """
        global event
        global semaphore
        event = threading.Event()
        semaphore = threading.Semaphore()

    def process_output_data(self, data):
        """
        Processes the output data. The convention for pytorch and nidaqmx is different. Therefore,
        the input to the device needs to be transposed before sending it to the device. Also the
        hardware does a current to voltage transformation to do the reading. For this,
        the output gets amplified. An amplification correction factor is applied to obtain the real
        current value again. This is done using the configs['driver']['amplification'] value. The
        function creates a numpy array from a list with dimensions (n,1) and multiplies
        it by the amplification of the device. It is transposed to enable the multiplication of
        multiple outputs by an array of amplification values.

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

        # If data has single dimension, create an extra dimension
        if len(data.shape) == 1:
            data = data[np.newaxis, :]

        # If there is a difference in points between read and write due to sampling frequencies, and there
        # is an average_io_point_difference flag set as True, the data is averaged
        if self.io_point_difference > 1 and self.configs['instruments_setup'][
                'average_io_point_difference']:
            data = np.mean(data.reshape(data.shape[0], -1,
                                        self.io_point_difference),
                           axis=2)

        return (data.T * self.configs["amplification"]).T

    def read_data(self, y):
        """
        Initializes the semaphore to read data from the device
        if the data cannot be read, a signal is sent to the signal handler which blocks the calling
        thread and closes the nidaqmx tasks

        Parameters
        ----------
        y : np.array
            Input data to be sent to the device.
            The data should have a shape of: (device_input_channel_no, data_point_no)
            Where device_input_channel_no is typically the number of activation
            electrodes of the DNPU.

        Returns
        -------
        list
            Output data that has been read from the device when receiving the input y.
        """
        global p
        p = Thread(target=self._read_data, args=(y, ))
        if not event.is_set():
            semaphore.acquire()
            p = Thread(target=self._read_data, args=(y, ))
            p.start()
            p.join()
            if self.data_results is None:
                print("Nothing could be read. Stopping program")
                self.os_signal_handler(None)
            semaphore.release()
        return self.data_results

    def set_io_configs(self, points_to_write, timeout=None):
        """
        Calculates and sets the I/O configuration variables related to the number of points
        the signal that is going to be writing and reading. This is only performed if there
        is a change with respect to the last number of points that were write/read.
        The calculation includes:
            - last_points_to_write_val: Number of points that were sent in the previos write attempt.
            - offsetted_points_to_write: Number of points with to be written with an extra offset
                                        that depends on the setup type. For cdaq, the default
                                        offset is 1 point, for nidaq, it is calculated from the
                                        sampling frequency.
            - set_sampling_frequencies: Sets the sampling frequencies of the activation and readout
                                        instruments.
            - timeout: Specifies the timeout for reading. Read below for more information.
            - points_to_read: Number of points that will be read given the number of points that
                              are written and the activation/readout frequency relationship.

        Parameters
        ----------
        points_to_write : (int,int)
            Number of points to be written.
        timeout: Specifies the amount of time in seconds to wait for samples to become
                available. If the time elapses, the method returns an error and any samples
                read before the timeout elapsed. The default timeout is 10 seconds. If you
                set timeout to nidaqmx.constants.WAIT_INFINITELY, the method waits
                indefinitely. If you set timeout to 0, the method tries once to read
                the requested samples and returns an error if it is unable to.
                By default, None, which calculates the timeout based on the frequency.


        """
        if self.last_points_to_write_val != points_to_write:
            self.last_points_to_write_val = points_to_write
            self.offsetted_points_to_write = points_to_write + self.configs[
                "offset"]
            self.points_to_read = self.tasks_driver.set_sampling_frequencies(
                self.configs["instruments_setup"]
                ["activation_sampling_frequency"],
                self.configs["instruments_setup"]
                ["readout_sampling_frequency"], self.offsetted_points_to_write)
            self.set_timeout(timeout)

    def set_timeout(self, timeout=None):
        """
        Updates the internal timeout value that will be used when reading the data.

        Parameters
        ----------
        timeout : int, optional
            Specifies the amount of time in seconds to wait for samples to become
            available. If the time elapses, the method returns an error and any samples
            read before the timeout elapsed. The default timeout is 10 seconds. If you
            set timeout to nidaqmx.constants.WAIT_INFINITELY, the method waits
            indefinitely. If you set timeout to 0, the method tries once to read
            the requested samples and returns an error if it is unable to.
            By default, None.
        """
        if timeout is None:
            timeout = self.offsetted_points_to_write * self.io_point_difference
            self.timeout = (math.ceil(timeout) + 10)  # Adds an extra second
        else:
            self.timeout = timeout

    def is_hardware(self):
        """
        Method to indicate whether this is a hardware processor. Returns
        True.

        Returns
        -------
        bool
            True
        """
        return True

    def _read_data(self, y):
        """
        Perfoms a series of security checks to the data, initialises the NI Tasks, and sends the
        data to the DNPU hardware. Returns the raw value obtained from the readout of the setup.

        Parameters
        -----------
        y : np.array
            Input data matrix to be sent to the device.
            The data should have a shape of: (device_input_channel_no, data_point_no)
            Where device_input_channel_no is typically the number of activation
            electrodes of the DNPU.
        Returns
        --------
        np.array
            Data read from the device

        """
        self.data_results = None
        self.read_security_checks(y)
        self.set_io_configs(y.shape[1])

        self.tasks_driver.write(y, self.configs["auto_start"])
        read_data = self.tasks_driver.read(self.points_to_read, self.timeout)
        self.tasks_driver.stop_tasks()

        self.data_results = read_data
        return read_data

    def read_security_checks(self, y):
        """
        This method reads the security checks from the input data, and makes sure that the input
        voltage does not go above certain threshhold.

        Parameters
        ----------
         y : np.array
            It represents the input data as matrix where the shape is defined by the "number of
            inputs to the device" times "input points that you want to input to the device".
        """
        for n, y_i in enumerate(y):
            assert all(y_i < INPUT_VOLTAGE_THRESHOLD), (
                f"Voltages in electrode {n} higher ({y_i.max()}) than the max."
                + f" allowed value ({INPUT_VOLTAGE_THRESHOLD} V)")
            assert all(y_i > -INPUT_VOLTAGE_THRESHOLD), (
                f"Voltages in electrode {n} lower ({y_i.min()}) than the min. "
                + "allowed value ({-INPUT_VOLTAGE_THRESHOLD} V)")
            assert (
                y_i[0] == 0.0
            ), f"First value of input stream in electrode {n} is non-zero ({y_i[0]})"
            assert (
                y_i[-1] == 0.0
            ), f"Last value of input stream in electrode {n} is non-zero ({y_i[-1]})"

    def close_tasks(self):
        """
        To close all NI tasks currently running on this device
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
        return self.configs["amplification"]

    def forward_numpy(self):
        """
        The forward function computes output numpy values from input numpy array.
        This is done to enable compatibility of the the model which is an nn.Module with numpy.
        This function will be overriden by the specific implementation in the CDAQ TO CDAQ or
        CDAQ TO NIDAQ setup.
        """
        pass

    def os_signal_handler(self, signum, frame=None):
        """
        Used to handle the termination of the read task in such a way that enables the last read
        call to the drivers to finish, and adequately closing the NI tasks afterwards.
        A handler for a particular signal, once set, remains installed until it is explicitly reset.

        More information can be found at:
        https://docs.python.org/3/library/signal.html

        Parameters
        ----------
        signum : int
            The signal number.
        frame : int, optional
            The current stack frame, by default None.
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
        Enables the OS signals by adding an a signal HandlerRoutine to support read/write in both
        linux and windows in Windows and Linux operating systems.
        """
        import win32api

        if sys.platform == "win32":
            win32api.SetConsoleCtrlHandler(self.os_signal_handler, True)
        else:
            signal.signal(signal.SIGTERM, self.os_signal_handler)
            signal.signal(signal.SIGINT, self.os_signal_handler)

    def disable_os_signals(self):
        """
        Disables the OS signals by removing the signal HandlerRoutine in the the win32 OS or
        ignoring the signal incase of other processors.
        """
        import win32api

        if sys.platform == "win32":

            win32api.SetConsoleCtrlHandler(None, True)
        else:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            signal.signal(signal.SIGINT, signal.SIG_IGN)
