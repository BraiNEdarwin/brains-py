"""
Class to manage torch variables and their data types. This class can be used to retrieve torch values and settings.
It can format data provided as a torch tensor,numpy array or an nn.Module object. It also handles the 
intialization of seed for generation of random values.
"""

import torch
import numpy as np
import random


class TorchUtils:
    """ A class to consistently manage declarations of torch variables for CUDA and CPU. """

    force_cpu = False

    @staticmethod
    def set_force_cpu(force: bool):
        """
        Enable setting the force CPU option for computers with an old CUDA version,
        where torch detects that there is cuda, but the version is too old to be compatible.

        Parameters
        ----------
        force : boolean
            True or false to set the force_cpu option to detect cuda.

        Example
        --------
        TorchUtils.set_force_cpu(True)

        """
        TorchUtils.force_cpu = force

    @staticmethod
    def get_device():
        """
        Consistently returns the accelerator type for the torch. The accelerator type of the device can be "cpu" or "cuda" depending on the version of the computer.

        Returns
        -------
        torch.device
            device type of torch tensor which can be "cpu" or "cuda" depending on computer version

        Example
        --------
        TorchUtils.get_device()
        """
        if torch.cuda.is_available() and not TorchUtils.force_cpu:
            return torch.device("cuda")
        return torch.device("cpu")

    @staticmethod
    def format(data, device=None, data_type=None):
        """
        Enables to create a torch variable with a consistent accelerator type and data type if a list or a numpy array is provided.
        If an exisiting torch tensor is provided, the function enables setting the data type and device consistently for all torch tensors
        For devices with more than one GPU, if an instance of an nn.Module is provided (for example - CustomModel,DNPU,Processor) this function helps to distribute the model and be more efficient

        Parameters
        ----------
        data :
            list/np.ndarray : list of data indices
            torch.Tensor :  inital torch tensor which has to be formatted
            nn.Module : model of an nn.Module object
        device : torch.device, optional
            device type of torch tensor which can be "cpu or "cuda" depending on computer version, by default None
        data_type : torch.dtype, optional
            desired data type of torch tensor, by default None

        Returns
        -------
        1. torch.tensor
            torch tensor either generated from python list or numpy array, or exisiting torch tensors formatted to given device and data type
        2. nn.Module
            if an nn.Module is given as an argument, a model of the nn.Module object distributed by DataParallel amongst the multiple GPUs is generated

        Example
        -------
        1.  data = [[1, 2]]
            tensor = TorchUtils.format(data, data_type=torch.float32)

        2.  tensor = torch.randn(2, 2)
            tensor = TorchUtils.format(tensor, data_type=torch.float64)

        3.  data = [[1, 2], [3, 4]]
            numpy_data = np.array(data)
            tensor = TorchUtils.format(numpy_data)

        4.  configs = {"optimizer" : "adam"}
            model = CustomModel()
            newmodel = format_model(model)

        """
        if device is None:
            device = TorchUtils.get_device()
        if data_type is None:
            data_type = torch.get_default_dtype()
        if isinstance(data, (list, np.ndarray, np.generic)):
            return torch.tensor(data, device=device, dtype=data_type)
        elif isinstance(data, torch.Tensor):
            return data.to(device=device, dtype=data_type)
        else:
            if torch.cuda.device_count() > 1 and not TorchUtils.force_cpu:
                data = torch.nn.DataParallel(data)
            data.to(TorchUtils.get_device())
            return data

    @staticmethod
    def to_numpy(data: torch.tensor):
        """
        Creates a numpy array from a torch tensor

        Parameters
        ----------
        data : torch.tensor
            torch tensor that needs to be formatted to a numpy array

        Returns
        -------
        np.array
            numpy array from given torch tensor

        Example
        -------
        tensor = torch.tensor([[1., -1.], [1., -1.]])
        numpy_data = TorchUtils.to_numpy(tensor)
        """
        if data.requires_grad:
            return data.detach().cpu().numpy()
        return data.cpu().numpy()

    @staticmethod
    def init_seed(seed=None, deterministic=False):
        """
        Sets the seed for generating random numbers.
        If the random seed is not reset, different numbers appear with every invocation

        Parameters
        ----------
        seed : int, optional
            value of seed, by default None
        deterministic : bool, optional
            if the random value should be deterministic, by default False

        Returns
        -------
        int
            value of seed

        Example
        -------
        TorchUtils.init_seed(0)
        random1 = np.random.rand(4)

        """
        if seed is None:
            seed = random.randint(0, (2 ** 32) - 1)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        return seed
