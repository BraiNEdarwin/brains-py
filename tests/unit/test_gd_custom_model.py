import unittest
import torch
from brainspy.algorithms.gd import train, default_train_step, default_val_step
from brainspy.utils.pytorch import TorchUtils
from brainspy.utils.performance.accuracy import zscore_norm
from brainspy.utils.performance.data import get_data
from tests.unit.testing_utils import DefaultCustomModel, CustomLogger, get_custom_model_configs


class GD_Test(unittest.TestCase):
    """
    Tests for Gradient Descent with a custom simulation model.
    """
    def get_train_parameters(self):
        """
        Generate some random train parameters for Gradient Descent
        """
        configs, model_data = get_custom_model_configs()
        model = DefaultCustomModel(configs, model_data['info'])
        model = TorchUtils.format(model)
        results = {}
        threshold = 20
        size = torch.randint(11, threshold, (1, 1)).item()
        results["inputs"] = TorchUtils.format(torch.rand((size, 2)))
        results["targets"] = TorchUtils.format(torch.randint(0, 2, (size, 1)))
        results["norm_inputs"] = zscore_norm(results["inputs"])
        dataloader = get_data(results, batch_size=512)
        dataloaders = [dataloader]
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
        configs = {}
        configs["epochs"] = 3
        configs["constraint_control_voltages"] = "regul"
        configs["regul_factor"] = 0.5

        return model, dataloaders, criterion, optimizer, configs

    def test_train(self):
        """
        Test for gradient descent with random train parameters
        """
        model, dataloaders, criterion, optimizer, configs = self.get_train_parameters(
        )
        logger = CustomLogger()
        dataloaders.append(dataloaders[0])
        try:
            model, results = train(model,
                                   dataloaders,
                                   criterion,
                                   optimizer,
                                   configs,
                                   save_dir='tests/data',
                                   return_best_model=True,
                                   logger=logger)
        except Exception as e:
            print(e)

            self.fail("Could not run Gradient Descent: ")
        else:
            self.assertTrue("performance_history" in results)

        configs = {}
        configs["epochs"] = 100
        configs["constraint_control_voltages"] = "clip"

        try:
            model, results = train(model, dataloaders, criterion, optimizer,
                                   configs)
        except (Exception):
            self.fail("Could not run Gradient Descent")
        else:
            self.assertTrue("performance_history" in results)

    def test_train_two_dataloaders(self):
        """
        Test for genetic algorithm with 2 dataloaders
        """
        model, dataloaders, criterion, optimizer, configs = self.get_train_parameters(
        )
        configs['constraint_control_voltages'] = 'clip'
        dataloaders.append(dataloaders[0])
        try:
            model, results = train(model, dataloaders, criterion, optimizer,
                                   configs)
        except Exception as e:
            print(e)

            self.fail("Could not run Gradient Descent: ")
        else:
            self.assertTrue("performance_history" in results)

        configs = {}
        configs["epochs"] = 3
        configs["constraint_control_voltages"] = "clip"

        try:
            model, results = train(model, dataloaders, criterion, optimizer,
                                   configs)
        except (Exception):
            self.fail("Could not run Gradient Descent")
        else:
            self.assertTrue("performance_history" in results)

    def test_invalid_model_type(self):
        """
        Wrong type of model raises an AssertionError
        """
        models = [{}, [1, 2, 3], "invalid type", 100, 5.5,
                  torch.nn.Linear(1, 1)]
        model, dataloaders, criterion, optimizer, configs = self.get_train_parameters(
        )
        logger = CustomLogger()
        for model in models:
            with self.assertRaises(AssertionError):
                train(model,
                      dataloaders,
                      criterion,
                      optimizer,
                      configs,
                      logger=logger)
            with self.assertRaises(AssertionError):
                default_train_step(model, 10, dataloaders, criterion,
                                   optimizer)
            with self.assertRaises(AssertionError):
                default_val_step(10,
                                 model,
                                 dataloaders,
                                 criterion,
                                 logger=logger)

    def test_invalid_dataloader_type(self):
        """
        Wrong type of dataloader raises an AssertionError
        """
        model, dataloader, criterion, optimizer, configs = self.get_train_parameters(
        )
        dataloaders = [dataloader[0], {}, 100, "invalid type"]
        for d in dataloaders:
            with self.assertRaises(AssertionError):
                train(model, d, criterion, optimizer, configs)
        dataloaders2 = [dataloader, {}, 100, "invalid type"]
        for d in dataloaders2:
            with self.assertRaises(AssertionError):
                default_train_step(model, 10, d, criterion, optimizer)
            with self.assertRaises(AssertionError):
                default_val_step(10, model, d, criterion)

    def test_invalid_optimizer_type(self):
        """
        Wrong type of optimizer raises an AssertionError
        """
        model, dataloaders, criterion, optimizer, configs = self.get_train_parameters(
        )
        optimizers = ["invalid type", 100, [1, 2, 3, 4, 5], {}]
        for o in optimizers:
            with self.assertRaises(AssertionError):
                train(model, dataloaders, criterion, o, configs)
            with self.assertRaises(AssertionError):
                default_train_step(model, 10, dataloaders, criterion, o)

    def test_invalid_criterion_type(self):
        """
        Wrong type of criterion raises an AssertionError
        """
        model, dataloaders, criterion, optimizer, configs = self.get_train_parameters(
        )
        criterions = ["invalid type", 100, [1, 2, 3, 4, 5], {}]
        for c in criterions:
            with self.assertRaises(AssertionError):
                train(model, dataloaders, c, optimizer, configs)
            with self.assertRaises(AssertionError):
                default_train_step(model, 10, dataloaders, c, optimizer)
            with self.assertRaises(AssertionError):
                default_val_step(10, model, dataloaders, c)

    def test_train_invalid_configs_type(self):
        """
        Wrong type of configs raises an AssertionError
        """

        model, dataloaders, criterion, optimizer, configs = self.get_train_parameters(
        )
        configs = ["invalid type", 100, [1, 2, 3, 4, 5]]
        for config in configs:
            with self.assertRaises(AssertionError):
                train(model, dataloaders, criterion, optimizer, config)

    def test_default_train_step(self):
        try:
            model, dataloaders, criterion, optimizer, configs = self.get_train_parameters(
            )
            for epoch in range(0, 3):
                model, running_loss = default_train_step(
                    model,
                    epoch,
                    dataloaders[0],
                    criterion,
                    optimizer,
                    logger=None,
                    constraint_control_voltages=configs[
                        'constraint_control_voltages'])
                self.assertIsNotNone(running_loss)
        except (Exception):
            self.fail("Could not perform default train step")

    def test_default_val_step(self):
        try:
            model, dataloaders, criterion, _, _ = self.get_train_parameters()
            logger = CustomLogger()

            running_loss = default_val_step(epoch=10,
                                            model=model,
                                            dataloader=dataloaders[0],
                                            criterion=criterion,
                                            logger=logger)
            self.assertIsNotNone(running_loss)
        except Exception:
            self.fail("Could not perform default val step")


if __name__ == "__main__":
    unittest.main()
