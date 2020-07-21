import torch
import numpy as np
import random


class TorchUtils:
    """ A class to consistently manage declarations of torch variables for CUDA and CPU. """
    force_cpu = False
    data_type = torch.float32

    @staticmethod
    def set_force_cpu(force):
        """ Enable setting the force CPU option for computers with an old CUDA version,
        where torch detects that there is cuda, but the version is too old to be compatible. """
        TorchUtils.force_cpu = force

    @staticmethod
    def set_data_type(data_type):
        """ Enable setting the force CPU option for computers with an old CUDA version,
        where torch detects that there is cuda, but the version is too old to be compatible. """
        TorchUtils.data_type = data_type

    @staticmethod
    def get_accelerator_type():
        """ Consistently returns the accelerator type for torch. """
        if torch.cuda.is_available() and not TorchUtils.force_cpu:
            return torch.device('cuda')
        return torch.device('cpu')

    # @staticmethod
    # def get_data_type():
    #     """It consistently returns the adequate data format for either CPU or CUDA.
    #     When the input force_cpu is activated, it will only create variables for the """
    #     if TorchUtils.get_accelerator_type() == 'cuda':
    #         return TorchUtils._get_cuda_data_type()
    #     return TorchUtils._get_cpu_data_type()

    # @staticmethod
    # def _get_cuda_data_type():
    #     if TorchUtils.data_type == 'float':
    #         return torch.cuda.FloatTensor
    #     if TorchUtils.data_type == 'long':
    #         return torch.cuda.LongTensor

    # @staticmethod
    # def _get_cpu_data_type():
    #     if TorchUtils.data_type == 'float':
    #         return torch.FloatTensor
    #     if TorchUtils.data_type == 'long':
    #         return torch.LongTensor
    #     # _ANS = type.__func__()
    #     # _ANS = data_type.__func__()

    @staticmethod
    def get_tensor_from_list(data, *args):
        """Enables to create a torch variable with a consistent accelerator type and data type."""
        if args:
            data_type = args[0]
        else:
            data_type = None
        return TorchUtils.format_tensor(torch.tensor(data), data_type=data_type)

    @staticmethod
    def format_tensor(tensor, data_type=None):
        """Enables setting the data type and device consistently for all torch.tensors"""
        if data_type is None:
            data_type = TorchUtils.data_type
        return tensor.to(device=TorchUtils.get_accelerator_type(), dtype=data_type)

    # _ANS = format_torch.__func__()

    @staticmethod
    def get_tensor_from_numpy(data):
        """Enables to create a torch variable from numpy with a consistent accelerator type and
        data type."""
        return TorchUtils.get_tensor_from_list(data)

    @staticmethod
    def get_numpy_from_tensor(data):
        if data.requires_grad:
            return data.detach().cpu().numpy()
        return data.cpu().numpy()

    @staticmethod
    def init_seed(seed=None, deterministic=False):
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
