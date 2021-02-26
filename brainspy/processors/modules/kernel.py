import torch
import numpy as np
from torch import nn
from brainspy.utils.pytorch import TorchUtils

from brainspy.processors.processor import Processor
from brainspy.processors.modules.vecbase import DNPUBase
from brainspy.processors.modules.bn import DNPU_BatchNorm
#from brainspy.utils.mappers import SimpleMapping
from brainspy.utils.electrodes import get_map_to_voltage_vars
import torch.nn.functional as F

class Kernel(nn.Module):
    def __init__(self, processor, inputs_list, in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0):
        super(Kernel,self).__init__()

        self.device_no = len(inputs_list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride

        self.amplitude = None
        self.offset = None

        if isinstance(processor, Processor):
            self.processor = processor
        else:
            self.processor = Processor(processor)  # It accepts initialising a processor as a dictionary
        ######### Set up node #########
        # Freeze parameters of node
        for params in self.processor.parameters():
            params.requires_grad = False

        self.indices_node = np.arange(len(self.processor.data_input_indices) + len(self.processor.control_indices))
        ######### set learnable parameters #########
        self.control_list = TorchUtils.get_tensor_from_list(self.set_controls(inputs_list), data_type=torch.int64).unsqueeze(0).repeat_interleave(self.in_channels, dim=0).unsqueeze(0).repeat_interleave(self.out_channels, dim=0)

        ######### Initialise data input ranges #########
        self.data_input_low = torch.stack([self.processor.processor.voltage_ranges[indx_cv, 0] for indx_cv in inputs_list]).unsqueeze(0).repeat_interleave(self.out_channels, dim=0)
        self.data_input_high = torch.stack([self.processor.processor.voltage_ranges[indx_cv, 1] for indx_cv in inputs_list]).unsqueeze(0).repeat_interleave(self.out_channels, dim=0)

        ###### Set everything as torch Tensors and send to DEVICE ######
        self.inputs_list = TorchUtils.get_tensor_from_list(inputs_list, data_type=torch.int64).unsqueeze(0).repeat_interleave(self.in_channels, dim=0).unsqueeze(0).repeat_interleave(self.out_channels, dim=0)
        # IndexError: tensors used as indices must be long, byte or bool tensors
        self.linear = torch.nn.Linear(5,1)

    def set_controls(self, inputs_list):
        control_list = [np.delete(self.indices_node, indx) for indx in inputs_list]
        control_low = [self.processor.processor.voltage_ranges[indx_cv, 0] for indx_cv in control_list]
        control_high = [self.processor.processor.voltage_ranges[indx_cv, 1] for indx_cv in control_list]
        self.control_low = torch.stack(control_low).squeeze().unsqueeze(0).repeat_interleave(self.in_channels, dim=0).unsqueeze(0).repeat_interleave(self.out_channels, dim=0)
        self.control_high = torch.stack(control_high).squeeze().unsqueeze(0).repeat_interleave(self.in_channels, dim=0).unsqueeze(0).repeat_interleave(self.out_channels, dim=0)
        # Sample control parameters
        # controls = [
        #     self.sample_controls(low, high)
        #     for low, high in zip(control_low, control_high)
        # ]
        
        # Register as learnable parameters
        self.all_controls = nn.Parameter(self.sample_controls(len(control_list[0]))) #@TODO: improve the sampling of cvs 


        return control_list

    def sample_controls(self, control_no):
        #@TODO: This code is very similar to that of add_transform. Create an additional function to avoid repeating code.
        data_input_range = [0,1]
        output_range = self.get_control_ranges()#.flatten(1,-1)
        min_input = data_input_range[0]
        max_input = data_input_range[1]
        input_range = torch.ones_like(output_range)
        input_range[0] *= min_input
        input_range[1] *= max_input
        amplitude, offset = get_map_to_voltage_vars(output_range[0],output_range[1],input_range[0],input_range[1])
        samples = torch.rand((self.out_channels,self.in_channels,self.device_no,control_no), device=TorchUtils.get_accelerator_type(), dtype=TorchUtils.get_data_type())
        return (amplitude * samples) + offset

    def add_transform(self,data_input_range, clip_input=False):
        output_range = self.get_input_ranges()#.flatten(1,-1)
        min_input = data_input_range[0]
        max_input = data_input_range[1]
        input_range = torch.ones_like(output_range)
        input_range[0] *= min_input
        input_range[1] *= max_input

        self.amplitude, self.offset = get_map_to_voltage_vars(output_range[0],output_range[1],input_range[0],input_range[1])
        # self.VariableRangeMapper()
        # self.transform = SimpleMapping(input_range=[-0.4242,2.8215], output_range=self.get_input_ranges().flatten(1,-1), clip_input=clip_input)

    def get_output_size(self):
        return int(((self.dim + (2*self.padding) - self.kernel_size)/self.stride ) + 1)

    # Evaluate node
    def forward(self, x):
        assert x.shape[2] == x.shape[3], "Different dimension shapes not supported"
        self.dim = x.shape[2]
        x = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding) # Unfold as in a regular convolution
        batch_size = x.shape[0]
        window_no = x.shape[-1] # Number of windows from the local receptive field after unfolding

        x = x.reshape(x.shape[0],self.in_channels,int(x.shape[1]/self.in_channels), x.shape[2]) # The window is divided by the number of input kernels 
        x = x.unsqueeze(1).repeat_interleave(self.out_channels,dim=1) # Repeat info that will be used for each DNPU kernel
        x = x.transpose(3, 4) #Transpose what will be inputed in the convolution by the number of windows. 

        x = x.reshape(x.shape[0],x.shape[1],x.shape[2],x.shape[3], self.device_no,self.inputs_list.shape[-1])# Divide what will be inputed in the convolution by the number of DNPUs. 

        if not (self.amplitude is None): #I
            amplitude = self.amplitude.unsqueeze(1).repeat_interleave(window_no,dim=1).unsqueeze(1).repeat_interleave(self.in_channels,dim=1).unsqueeze(0).repeat_interleave(batch_size,dim=0)
            offset = self.offset.unsqueeze(1).repeat_interleave(window_no,dim=1).unsqueeze(1).repeat_interleave(self.in_channels,dim=1).unsqueeze(0).repeat_interleave(batch_size,dim=0)
            x = (x * amplitude) + offset

        #x = torch.transpose(x, 2,3).flatten(0,2).reshape(-1,self.inputs_list.shape[-1])
        #x = x.reshape(-1,self.inputs_list.shape[-1]) 
        # Reshape input and expand controls
        controls = self.all_controls.unsqueeze(2).repeat_interleave(window_no,dim=2).unsqueeze(0).repeat_interleave(batch_size,dim=0)
        last_dim = len(controls.shape) - 1

        # Expand indices according to batch size
        input_indices = self.inputs_list.unsqueeze(2).repeat_interleave(window_no,dim=2).unsqueeze(0).repeat_interleave(batch_size,dim=0)
        control_indices = self.control_list.unsqueeze(2).repeat_interleave(window_no,dim=2).unsqueeze(0).repeat_interleave(batch_size,dim=0)

        # Create input data and order it according to the indices
        indices = torch.cat((input_indices,control_indices),dim=last_dim)
        data = torch.cat((x,controls),dim=last_dim)
        data = torch.gather(data,last_dim,indices)
        self.data_dim = data.shape
        data = data.reshape(-1,data.shape[-1])

        result = self.processor.processor(data).squeeze()
        result = result.reshape(controls.shape[0],controls.shape[1],controls.shape[2],controls.shape[3],controls.shape[4]).sum(dim=4).sum(dim=2) # Sum extra dimensions
        #result = result.reshape(controls.shape[0],controls.shape[1],controls.shape[2],controls.shape[3],controls.shape[4])
        #result = result.flatten(0,3)
        #result = self.linear(result)
        #result = result.sum(dim=1)
        #result = result.reshape(controls.shape[0],controls.shape[1]*controls.shape[2],controls.shape[3])
        result = torch.nn.functional.fold(result,kernel_size=1, output_size=self.get_output_size())
        return  result # * self.node.amplification

    def reset(self):
        raise NotImplementedError("Resetting controls not implemented!!")
        # for k in range(len(self.control_low)):
        #     # print(f'    resetting control {k} between : {self.control_low[k], self.control_high[k]}')
        #     self.controls.data[:, k].uniform_(self.control_low[k], self.control_high[k])

    def regularizer(self):
        if 'control_low' in dir(self) and 'control_high' in dir(self):
            return 0
        else:
            assert any(
                self.control_low.min(dim=0)[0] < 0
            ), "Min. Voltage is assumed to be negative, but value is positive!"
            assert any(
                self.control_high.max(dim=0)[0] > 0
            ), "Max. Voltage is assumed to be positive, but value is negative!"
            buff = 0.0
            for i, p in enumerate(self.all_controls):
                buff += torch.sum(
                    torch.relu(self.control_low[i] - p)
                    + torch.relu(p - self.control_high[i])
                )
            return buff

    def hw_eval(self, hw_processor_configs):
        self.processor.hw_eval(hw_processor_configs)

    def is_hardware(self):
        return self.processor.is_hardware

    def get_clipping_value(self):
        return self.processor.get_clipping_value()

    def get_input_ranges(self):
        return torch.stack((self.data_input_low,self.data_input_high))

    def get_control_ranges(self):
        return torch.stack((self.control_low,self.control_high))  # Total Dimensions 3: Dim 0: 0=min volt range1=max volt range, Dim 1: Index of node, Dim 2: Index of electrode

    def get_control_voltages(self):
        return torch.vstack([cv.data.detach() for cv in self.all_controls]).flatten()

    def set_control_voltages(self, control_voltages):
        with torch.no_grad():
            #bias = bias.unsqueeze(dim=0)
            assert (
                self.all_controls.shape == control_voltages.shape
            ), "Control voltages could not be set due to a shape missmatch with regard to the ones already in the model."
            self.bias = torch.nn.Parameter(TorchUtils.format_tensor(control_voltages))