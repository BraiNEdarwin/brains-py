import torch.nn as nn

from brainspy.processors.dnpu import DNPU
from brainspy.processors.processor import Processor


class DNPUBatchNorm(DNPU):
    """
    A child of brainspy.processors.dnpu.DNPU class that adds a batch normalisation layer after the
    output. for adding a batch normalisation layer after the output of a DNPU.

    More information about batch normalisation can be found in:
    https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html

    Attributes:
    bn : torch.nn.Module
        A batch normalisation module that is an instance of a torch.nn.Module.
    verbose : bool
        Indicate whether to print certain steps.
    raw_model : nn.Sequential
        Torch object containing the layers and activations of the network.
    """
    def __init__(self,
                 processor: Processor,
                 data_input_indices: list,
                 forward_pass_type: str = 'vec',
                 affine=False,
                 track_running_stats=True,
                 momentum=0.1,
                 eps=1e-5,
                 custom_bn=nn.BatchNorm1d):
        """
        Initialises the super class and the batch normalisation module, according to the batch norm
        parameters given (affine, track_running_stats, momentum, eps, custom_bn).

        More information about batch normalisation can be found in:
        https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html

        Attributes:
        processor : brainspy.processors.processor.Processor
            An instance of a Processor, which can hold a DNPU model or a driver connection to the
            DNPU hardware.
        data_input_indices: list
            Specifies which electrodes are going to be used for inputing data. The reminder of the
            activation electrodes will be automatically selected as control electrodes. The list
            should have the following shape (dnpu_node_no,data_input_electrode_no). The minimum
            dnpu_node_no should be 1, e.g., data_input_indices = [[1,2]]. When specifying more than
            one dnpu node in the list, the module will simulate, in time-multiplexing,
            as if there was a layer of DNPU devices. Fore example, for an 8 electrode DNPU device
            with a single readout electrode and 7 activation electrodes, when
            data_input_indices = [[1,2],[1,3],[3,4]], it will be considered that there are 3 DNPU
            devices, where the first DNPU device will use the data input electrodes
            1 and 2, the second DNPU device will use data input electrodes 1 and 3 and the third
            DNPU device will use data input electrodes 3 and 4. Also, the first DNPU device will
            have electrodes 0, 3, 4, 5, and 6 defined as control electrodes. The second DNPU device
            will have electrodes 0,2,4,5, and 6 defined as control electrodes. The third DNPU device
            will have electrodes 0,1,2,5, and 6 defined as control electrodes. More information
            about what activation, readout, data input and control electrodes are can be found at
            the wiki: https://github.com/BraiNEdarwin/brains-py/wiki/A.-Introduction
        forward_pass_type : str
            It indicates if the forward pass for more than one DNPU devices on time-multiplexing
            will be executed using vectorisation or a for loop. The available options are 'vec' or
            'for'. By default it uses the vectorised version.
        affine : A boolean value that when set to True, this module has learnable affine parameters.
                 By default is set to False, in order to save using extra parameters.
        track_running_stats : bool
            A boolean value that when set to True, this module tracks the running mean and variance,
            and when set to False, this module does not track such statistics, and initializes
            statistics buffers running_mean and running_var as None. When these buffers are None,
            this module always uses batch statistics in both training and eval modes. Default: True
        momentum : float
            The value used for the running_mean and running_var computation. Can be set to None for
            cumulative moving average (i.e. simple average). Default: 0.1
        eps : float
            A value added to the denominator for numerical stability. Default: 1e-5
        custom_bn : torch.nn.Module
            A batch normalisation module that is an instance of a torch.nn.Module. By default
            torch.nn.BatchNorm1d
        """
        super(DNPUBatchNorm,
              self).__init__(processor,
                             data_input_indices,
                             forward_pass_type=forward_pass_type)
        self.bn = custom_bn(self.get_node_no(),
                            affine=affine,
                            track_running_stats=track_running_stats,
                            momentum=momentum,
                            eps=eps)

    def forward(self, x):
        """  Run a forward pass through the processor, including any time-multiplexing modules that
        are declared to be measured in the same layer. After getting the output from the processor
        the output is passed through the batch normalisation layer.

        Parameters
        ----------
        x : torch.Tensor
            Input data.

        Returns
        -------
        torch.Tensor
            Output data.
        """
        self.dnpu_output = self.forward_pass(x)
        self.batch_norm_output = self.bn(self.dnpu_output)
        return self.batch_norm_output

    def get_logged_variables(self):
        """ Get the otuput results from each layer from the last forward pass.

            Returns
            -------
            dict
                Dictionary containing the output from the last forward pass as a dictionary.
                    c_dnpu_output: Output of the dnpu / dnpu layer
                    d_batch_norm_output: Output of the batch norm layer
            """
        return {
            "c_dnpu_output": self.dnpu_output.clone().detach(),
            "d_batch_norm_output": self.batch_norm_output.clone().detach(),
        }
