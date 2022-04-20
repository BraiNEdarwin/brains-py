"""
Class to consistently manage torch variables and their data types. It can format
data provided as a torch tensor,numpy array or an nn.Module object. It also han-
dles the intialization of seed for generation of random values.
"""

import torch
import numpy as np
import random


class TorchUtils:
    """ Consistently manage data movement and formatting of pytorch tensors and torch.nn.Module
    instances. This includes data movement to/from CUDA and CPU, as well as transformations to/
    from lists and numpy arrays. It also enables to manage the seeds for reproducibility purpo-
    ses.
        More information can be found at:
        https://pytorch.org/docs/stable/torch.html
        https://pytorch.org/docs/stable/notes/randomness.html
    """

    force_cpu = False

    @staticmethod
    def set_force_cpu(force: bool):
        """
        Facilitates running all tensors on the CPU even when having a GPU with cuda correctly
        installed. Ideal for GPUs that do not have enough memory to run certain experiments or
        for computers with or with older GPUs. Also, for cases where torch detects that there
        is cuda, but the version is too old to be compatible.

        Parameters
        ----------
        force : boolean
            True or false to set the force_cpu option to detect cuda.

        Example
        --------
        TorchUtils.set_force_cpu(True)

        """
        assert (type(force) == bool)
        TorchUtils.force_cpu = force

    @staticmethod
    def get_device():
        """
        Consistently returns the device type for the torch. The device type can be "cpu" or "cuda",
        depending on if the computer has a GPU with a cuda installation, and on the variable
        force_cpu. This method does not support the management of multiple GPUs.

        Returns
        -------
        torch.device
            Device type of torch tensor which can be "cpu" or "cuda" depending on computer version.

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
        If a list or a numpy array is provided, it enables to create a torch variable with a
        consistent accelerator type and data type. If an exisiting torch tensor is provided,
        the function enables setting the data type and device consistently for all torch tensors.
        For devices with more than one GPU, if an instance of an nn.Module is provided, for example
        CustomModel,DNPU or Processor, this function helps to distribute the model.

        Parameters
        ----------
        data : Data to be formatted. It can be one of these data types:
            list : list of data indices
            np.ndarray : list of data indices
            torch.Tensor :  inital torch tensor which has to be formatted
            nn.Module : model of an nn.Module object
        device : torch.device, optional
            Device type of torch tensor which can be "cpu or "cuda" depending on computer version.
            When set to none, it will take the default value from the global variable force_cpu of
            this class. By default device is set to None.
        data_type : torch.dtype, optional
            desired data type of torch tensor, by default None
            When set to None, it will take the default data type from pytorch. This datatype can be
            changed using the torch.set_default_dtype. More info on this method at:
            https://pytorch.org/docs/stable/generated/torch.set_default_dtype.html

        Returns
        -------
        1. torch.tensor
            Torch tensor either generated from python list or numpy array, or exisiting torch
            tensors formatted to given device and data type.
        2. nn.Module
            If an nn.Module is given as an argument, a model of the nn.Module object distributed by
            DataParallel amongst the multiple GPUs is generated.

        Examples
        -------
        1.  data = [[1, 2]]
            tensor = TorchUtils.format(data, device=torch.device('cpu'))

        2.  tensor = torch.randn(2, 2)
            tensor = TorchUtils.format(tensor, data_type=torch.float64)

        3.  data = [[1, 2], [3, 4]]
            numpy_data = np.array(data)
            tensor = TorchUtils.format(numpy_data)

        4.  model = CustomModel() # Where CustomModel is an instance of torch.nn.Module
            newmodel = format_model(model)

        """
        if device is None:
            device = TorchUtils.get_device()
        if data_type is None:
            data_type = torch.get_default_dtype()
        if isinstance(data, (np.ndarray, np.generic)):
            return torch.tensor(data, device=device, dtype=data_type)
        elif isinstance(data, torch.Tensor):
            return data.to(device=device, dtype=data_type)
        elif isinstance(data, list):
            return torch.tensor(np.array(data), device=device, dtype=data_type)
        elif isinstance(data, torch.nn.Module):
            if torch.cuda.device_count() > 1 and not TorchUtils.force_cpu:
                data = torch.nn.DataParallel(data)
            data.to(TorchUtils.get_device())
            return data
        else:
            raise TypeError(
                "Input type not recognised. " +
                "Supported types are: lists, Numpy arrays, Pytorch tensors and modules."
            )

    @staticmethod
    def to_numpy(data: torch.Tensor):
        """
        Transforms torch tensor into a numpy array, detatching it first, and
        moving the data to the cpu (if needed). The aim is to simplify the lines
        of code required for this purpose with the original pytorch library.

        Parameters
        ----------
        data : torch.tensor
            Torch tensor that needs to be formatted to a numpy array.

        Returns
        -------
        np.array
            Numpy array from given torch tensor.

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
        It enables to set a fixed seed, or retrieve the current seed used for generating random
        numbers. If the random seed is not reset, different numbers appear with every invocation.
        It can be used to reproduce results.
        More information at: https://pytorch.org/docs/stable/notes/randomness.html

        Parameters
        ----------
        seed : int, optional
            Value of seed, by default None.
        deterministic : bool, optional
            If the random value should be deterministic, by default False.

        Returns
        -------
        int
            Value of the seed that is being used.

        Example
        -------
        TorchUtils.init_seed(0)
        random1 = np.random.rand(4)

        """
        if seed is None:
            seed = random.randint(0, (2**32) - 1)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        return seed
