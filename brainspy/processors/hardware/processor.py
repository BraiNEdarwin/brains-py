""" The accelerator class enables to statically access the accelerator
(CUDA or CPU) that is used in the computer. The aim is to support both platforms seemlessly. """

import torch
from torch import nn
from brainspy.utils.manager import get_driver
from brainspy.utils.pytorch import TorchUtils
from brainspy.utils.waveform import WaveformManager
from brainspy.processors.simulation.processor import SurrogateModel


class HardwareProcessor(nn.Module):
    """
    The TorchModel class is used to manage together a torch model and its state dictionary. The usage is expected to be as follows
    mymodel = TorchModel()
    mymodel.load_model('my_path/my_model.pt')
    mymodel.model
    """

    # TODO: Automatically register the data type according to the configurations of the amplification variable of the  info dictionary
    def __init__(self, configs, logger=None):
        """
        Method to o intialise the hardware processor

        Parameters
        ----------
        configs : dict
        Data key,value pairs required in the configs to initialise the hardware processor :

            processor_type : "simulation_debug" - Processor type to initialize a hardware processor
            data:
                waveform:
                    plateau_length: int - A plateau of at least 3 is needed to train the perceptron (That requires at least 10 values (3x4 = 12)).
                    slope_length : int - Length of the slope of a waveform
                activation_electrode_no: int - It specifies the number of activation electrodes. Only required for simulation mode
            driver:
                amplification: float - To set the amplification value of the voltages
                output_clipping_range: [float,float] - To clip the output voltage if it goes above maximum

        logger : [type], optional
            [description], by default None
        """
        super(HardwareProcessor, self).__init__()
        # @TODO: check if all the configs inputed to the hardware processor ar really needed somewhere
        # Should only configs['driver'] be passed to the driver?
        self.driver = get_driver(configs)
        if configs["processor_type"] == "simulation_debug":
            self.voltage_ranges = self.driver.voltage_ranges
        else:
            self.voltage_ranges = TorchUtils.format(self.driver.voltage_ranges)
        self.waveform_mgr = WaveformManager(configs["data"]["waveform"])
        self.logger = logger
        # TODO: Manage amplification from this class
        self.amplification = configs["driver"]["amplification"]
        self.clipping_value = [
            configs["driver"]["output_clipping_range"][0] * self.amplification,
            configs["driver"]["output_clipping_range"][1] * self.amplification,
        ]
        self.electrode_no = configs["data"]["activation_electrode_no"]

    def forward(self, x):
        with torch.no_grad():
            x, mask = self.waveform_mgr.plateaus_to_waveform(x, return_pytorch=False)
            output = self.forward_numpy(x)
            if self.logger is not None:
                self.logger.log_output(x)
        return TorchUtils.format(output[mask])

    def forward_numpy(self, x):
        return self.driver.forward_numpy(x)

    def reset(self):
        self.driver.reset()

    def close(self):
        if "close_tasks" in dir(self.driver):
            self.driver.close_tasks()
        else:
            print("Warning: Driver tasks have not been closed.")

    def is_hardware(self):
        return self.driver.is_hardware()

    def get_electrode_no(self):
        return self.electrode_no
