import torch
import numpy as np
import random


class TorchUtils:
    """ A class to consistently manage declarations of torch variables for CUDA and CPU. """

    force_cpu = False
    data_type = torch.get_default_dtype()

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
    def set_data_type(data_type: torch.dtype):
        """
        The function sets the data_type value to to a new datatype

        Parameters
        ----------
        data_type : torch.dtype
            desired data type of torch tensor

        Example
        -------
        TorchUtils.set_data_type(torch.float64)
        """
        TorchUtils.data_type = data_type

    @staticmethod
    def get_data_type():
        """The function gets the current value of the data_type variable

        Returns
        -------
        data_type
            current data type of torch tensor

        Example
        --------
        TorchUtils.get_data_type()
        """
        return TorchUtils.data_type

    @staticmethod
    def get_device():
        """Consistently returns the accelerator type for torch.

        Returns
        -------
        torch.device
            device type of torch tensor which can be "cpu or "cuda" depending on computer version

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
        If an exisiting torch tensor is provided, the function enables setting the data type and device consistently for all torch.tensors

        Parameters
        ----------
        data :
            list/np.ndarray : list of data indices
            torch.Tensor :  inital torch tensor which has to be formatted
        device : torch.device, optional
            device type of torch tensor which can be "cpu or "cuda" depending on computer version, by default None
        data_type : torch.dtype, optional
            desired data type of torch tensor, by default None

        Returns
        -------
        torch.tensor
            torch tensor either generated from python list or numpy array, or exisiting torch tensors formatted to given device and data type

        Example
        -------
        1.   data = [[1, 2]]
             tensor = TorchUtils.format(data, data_type=torch.float32)

        2.   tensor = TorchUtils.format(data, data_type=torch.float64)

        3.   data = [[1, 2], [3, 4]]
             numpy_data = np.array(data)
             tensor = TorchUtils.format(numpy_data)

        """
        if isinstance(data, (list, np.ndarray, np.generic)):
            if device is None:
                device = TorchUtils.get_device()
            if data_type is None:
                data_type = TorchUtils.get_data_type()
            return torch.tensor(data, device=device, dtype=data_type)
        else:
            if device is None:
                device = TorchUtils.get_device()
            if data_type is None:
                data_type = TorchUtils.get_data_type()
            return data.to(device=device, dtype=data_type)

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

    @staticmethod
    def format_model(model):
        """
        For devices with more than one GPU, this function helps to distribute the model and be more efficient

        Parameters
        ----------
        model : nn.Module
            model of an nn.Module object

        Returns
        -------
        nn.Module
            model of an nn.Module object distributed by DataParallel amongst the multiple GPUs

        Example
        --------
        configs = {"optimizer" : "adam"}
        model = CustomModel()
        newmodel = format_model(model)

        """
        if torch.cuda.device_count() > 1 and not TorchUtils.force_cpu:
            model = torch.nn.DataParallel(model)
        model.to(TorchUtils.get_device())
        return model
