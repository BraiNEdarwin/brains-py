import datetime as d
import tracemalloc
import unittest
import HtmlTestRunner
import numpy as np
import torch
import yaml
from brainspy.utils.io import *
from brainspy.utils.loader import *
from brainspy.utils.manager import *
from brainspy.utils.pytorch import TorchUtils
from numpy import asarray, load
from brainspy.processors.processor import Processor
from brainspy.algorithms.modules.optim import GeneticOptimizer
from brainspy.utils.transforms import (
    DataToTensor,
    DataToVoltageRange,
    DataPointsToPlateau,
)
from torchvision import transforms
from brainspy.processors.dnpu import DNPU
from torch.utils.data import Dataset
from brainspy.processors.simulation.processor import SurrogateModel
from brainspy.processors.hardware.drivers.cdaq import CDAQtoCDAQ
from brainspy.processors.hardware.drivers.nidaq import CDAQtoNiDAQ
from tests.unit.utils.testfiles.dataset import BooleanGateDataset


class ManagerTest(unittest.TestCase):

    """
    Tests for the manager.py class.
    To run the test, change the path variable in the places indicated below.
    Most of the tests use yaml files which can be manipulated and require a model.pt file whose path should be specified in the yaml file.
    """

    def __init__(self, test_name):
        super(ManagerTest, self).__init__()
        # self.path = os.path.join(str(os.getcwd()), "tests/unit/utils/testfiles")
        self.path = "C:/users/humai/brains-py/tests/unit/utils/testfiles"  # Enter path to the testfiles directory

    def test_get_criterion(self):
        """
        Test for the get_criterion() method to return a valid fitness function.
        This method is tested with the test_torch.yaml file that defines a fitness criterion.
        This fitness function is tested with 2 arbitrary torch tensors namely "predictions" and "targets".
        """
        configs = load_configs(self.path + "/test_torch.yaml")
        criterion = get_criterion(configs["algorithm"])
        targets = torch.tensor(
            [
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [1.0],
                [1.0],
                [1.0],
            ]
        )
        predictions = torch.tensor(
            np.array(
                [
                    [-0.6571],
                    [-1.7871],
                    [-1.1129],
                    [-0.7235],
                    [0.3469],
                    [-0.5552],
                    [-0.3134],
                    [0.5727],
                    [1.4812],
                    [0.3388],
                    [0.9651],
                    [0.9615],
                ]
            )
        )
        corsigfit = criterion(predictions, targets)
        assert isinstance(corsigfit, torch.Tensor)
        val = torch.tensor(0.4282, dtype=torch.float64)
        self.assertEqual(corsigfit.shape, val.shape)

    def test_get_optimizer(self):
        """
        Test for the get_optimizer() method which returns an optimizer object based on the configs defined in the test_torch.yaml file
        """
        configs = load_configs(self.path + "/test_torch.yaml")
        model = Processor(configs)
        optim = get_optimizer(model, configs["algorithm"])
        assert isinstance(optim, GeneticOptimizer)

    def test_get_adam(self):
        """
        Test for the get_adam() method which returns an Adam optimizer object based on the configs defined in the test_adam.yaml file
        """
        configs = load_configs(self.path + "/test_adam1.yaml")
        model = Processor(configs)
        optim = get_optimizer(model, configs["algorithm"])
        op = optim.state_dict()
        self.assertEqual(str(op["param_groups"][0]["eps"]), "1e-08")

    def test_get_algorithm(self):
        """
        Test for the get_algorithm() method which returns a train function based on the configs defined in the test_torch.yaml file
        This train function is is tested with a sample dataset which returns a trained model
        """
        configs = load_configs(self.path + "/test_torch.yaml")
        algorithm = get_algorithm(configs["algorithm"])
        model = train(algorithm, configs)
        self.assertTrue(model is not None)

    def test_get_driver(self):
        """
        Test for the get_driver() method which returns a driver object based on the configurations dictionary
        """
        configs = load_configs(self.path + "/test_torch.yaml")
        driver = get_driver(configs)
        assert isinstance(driver, SurrogateModel)

        # Test For Hardware Processor

        # configs["processor_type"] = "cdaq_to_cdaq"
        # configs["tasks_driver_type"] = "local"
        # driver = get_driver(configs)
        # assert isinstance(driver, CDAQtoCDAQ)
        # configs["processor_type"] = "cdaq_to_nidaq"
        # driver = get_driver(configs)
        # assert isinstance(driver, CDAQtoNiDAQ)

    def runTest(self):
        self.test_get_criterion()
        self.test_get_optimizer()
        self.test_get_adam()
        self.test_get_algorithm()
        self.test_get_driver()


def train(algorithm, configs):
    """
    Function to test the train function which is obtained from the get_algorithm() method
    """
    path = "C:/users/humai/brains-py/tests/unit/utils/testfiles"  # Enter path to save train results
    waveform_transforms = transforms.Compose(
        [DataPointsToPlateau(configs["processor"]["data"]["waveform"])]
    )
    V_MIN = [-1.2, -1.2]
    V_MAX = [0.6, 0.6]
    data_transforms = transforms.Compose(
        [
            DataToVoltageRange(V_MIN, V_MAX, -1, 1),
            DataToTensor(device=torch.device("cpu")),
        ]
    )
    criterion = get_criterion(configs["algorithm"])
    model = DNPU(configs)
    optimizer = get_optimizer(model, configs["algorithm"])
    dataset = BooleanGateDataset(
        target=np.array([0, 0, 0, 1]), transforms=data_transforms
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=len(dataset),
        shuffle=True,
        pin_memory=configs["data"]["pin_memory"],
    )
    return algorithm(
        model,
        (loader, None),
        criterion,
        optimizer,
        configs["algorithm"],
        waveform_transforms=waveform_transforms,
        logger=None,
        save_dir=path,
    )


if __name__ == "__main__":
    unittest.main(
        testRunner=HtmlTestRunner.HTMLTestRunner(
            output="C:/users/humai/brains-py/tests/unit/utils/testfiles/"  # Enter the path where you want to save the results
        )
    )
