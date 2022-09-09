import unittest
import warnings
import numpy as np
import torch
from brainspy.algorithms.ga import train, evaluate_population
from brainspy.utils.pytorch import TorchUtils
from brainspy.utils.performance.accuracy import zscore_norm
from brainspy.utils.performance.data import get_data
from brainspy.algorithms.ga import GeneticOptimizer
from tests.test_utils import DefaultCustomModel, get_custom_model_configs, is_hardware_fake, fake_criterion


class GA_Test_SurrogateModel(unittest.TestCase):
    """
    Tests for the Genetic Algorithm with a Simulation Processor
    """
    def get_train_parameters(self):
        """
        Generate some random train parameters for Genetic algorithm
        """
        configs, model_data = get_custom_model_configs()
        model = DefaultCustomModel(configs, model_data['info'])
        model = TorchUtils.format(model)
        results = {}
        threshhold = 20
        size = torch.randint(11, threshhold, (1, 1)).item()
        results["inputs"] = TorchUtils.format(torch.rand((size, 2)))
        results["targets"] = TorchUtils.format(torch.randint(0, 2, (size, 1)))
        results["norm_inputs"] = zscore_norm(results["inputs"])
        dataloader = get_data(results, batch_size=256)
        dataloaders = [dataloader]
        criterion = torch.nn.MSELoss()
        optimizer = GeneticOptimizer(gene_ranges=TorchUtils.format(
            torch.tensor([[-1.2, 0.6], [-1.2, 0.6], [-1.2, 0.6], [-1.2, 0.6],
                          [-1.2, 0.6]])),
                                     partition=[4, 22],
                                     epochs=3)
        configs = {}
        configs["epochs"] = 100
        configs["stop_threshold"] = 0.5
        return model, dataloaders, criterion, optimizer, configs

    def test_train(self):
        """
        Test for genetic algorithm with random inputs using a surrogate model
        """
        model, dataloaders, criterion, optimizer, configs = self.get_train_parameters(
        )
        try:
            model, results = train(model,
                                   dataloaders,
                                   criterion,
                                   optimizer,
                                   configs,
                                   save_dir='tests/data')
        except (Exception):
            self.fail("Could not run Genetic Algorithm")
        else:
            self.assertTrue("best_result_index" in results)
            self.assertTrue("genome_history" in results)
            self.assertTrue("performance_history" in results)
            self.assertTrue("correlation_history" in results)
            self.assertTrue("best_output" in results)

    def test_train_low_threshold(self):
        """
        Test for genetic algorithm with random inputs using a surrogate model
        """
        model, dataloaders, criterion, optimizer, configs = self.get_train_parameters(
        )
        try:
            configs['stop_threshold'] = -np.inf
            model.is_hardware = is_hardware_fake
            model, results = train(model,
                                   dataloaders,
                                   criterion,
                                   optimizer,
                                   configs,
                                   save_dir='tests/data')
        except (Exception):
            self.fail("Could not run Genetic Algorithm")
        else:
            self.assertTrue("best_result_index" in results)
            self.assertTrue("genome_history" in results)
            self.assertTrue("performance_history" in results)
            self.assertTrue("correlation_history" in results)
            self.assertTrue("best_output" in results)

    def test_train_invalid_model(self):
        """
        Running genetic algorithm on an external pytorch model raises an Exception
        """
        with self.assertRaises(Exception):
            model = torch.nn.Linear((1, 1))
            dataloaders, criterion, optimizer, configs = self.get_train_parameters(
            )
            train(model, dataloaders, criterion, optimizer, configs)

    def test_train_invalid_model_type(self):
        """
        Wrong type of model raises an AssertionError
        """
        models = [{}, [1, 2, 3], "invalid type", 100, 5.5]
        model, dataloaders, criterion, optimizer, configs = self.get_train_parameters(
        )
        for model in models:
            with self.assertRaises(AssertionError):
                train(model, dataloaders, criterion, optimizer, configs)

    def test_train_invalid_dataloader_type(self):
        """
        Wrong type of dataloader raises an AssertionError
        """
        model, dataloader, criterion, optimizer, configs = self.get_train_parameters(
        )
        dataloaders = [dataloader[0], {}, 100, "invalid type"]
        for d in dataloaders:
            with self.assertRaises(AssertionError):
                train(model, d, criterion, optimizer, configs)

    def test_train_invalid_criterion_type(self):
        """
        Wrong type of criterion raises an AssertionError
        """
        model, dataloaders, criterion, optimizer, configs = self.get_train_parameters(
        )
        criterions = ["invalid type", 100, [1, 2, 3, 4, 5], {}]
        for c in criterions:
            with self.assertRaises(AssertionError):
                train(model, dataloaders, c, optimizer, configs)

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

    def test_evaluate_population(self):
        """
        Test for the evaluate_population method
        """
        try:
            model, dataloaders, criterion, optimizer, configs = self.get_train_parameters(
            )
            inputs, targets = dataloaders[0].dataset[:]
            inputs, targets = TorchUtils.format(inputs), TorchUtils.format(
                targets)
            outputs, criterion_pool = evaluate_population(
                inputs, targets, optimizer.pool, model, criterion)
        except (Exception):
            self.fail("Could not execute function for the surrogate model")
        else:
            self.assertTrue(outputs is not None)
            self.assertIsInstance(outputs, torch.Tensor)
            self.assertTrue(criterion_pool is not None)
            self.assertIsInstance(criterion_pool, torch.Tensor)

    def test_evaluate_population_negatives(self):
        """
        Test for the evaluate_population method
        """
        # try:
        model, dataloaders, criterion, optimizer, configs = self.get_train_parameters(
        )
        inputs, targets = dataloaders[0].dataset[:]
        inputs, targets = TorchUtils.format(inputs), TorchUtils.format(targets)
        outputs, criterion_pool = evaluate_population(inputs, targets,
                                                      optimizer.pool, model,
                                                      fake_criterion)
        print('a')
        # except (Exception):
        #     self.fail("Could not execute function for the surrogate model")
        # else:
        #     self.assertTrue(outputs is not None)
        #     self.assertIsInstance(outputs, torch.Tensor)
        #     self.assertTrue(criterion_pool is not None)
        #     self.assertIsInstance(criterion_pool, torch.Tensor)

    def test_evaluate_population_fail(self):
        """
        Invalid type for inputs or targets raises an AssertionError
        """
        inputs = ["invalid type", 100, [1, 2, 3, 4, 5], {}]
        targets = ["invalid type", 100, [1, 2, 3, 4, 5], {}]
        model, dataloaders, criterion, optimizer, configs = self.get_train_parameters(
        )
        input, target = dataloaders[0].dataset[:]
        input, target = TorchUtils.format(input), TorchUtils.format(target)
        model, dataloaders, criterion, optimizer, configs = self.get_train_parameters(
        )
        for i in inputs:
            with self.assertRaises(AssertionError):
                evaluate_population(i, target, optimizer.pool, model,
                                    criterion)
        for t in targets:
            with self.assertRaises(AssertionError):
                evaluate_population(input, t, optimizer.pool, model, criterion)


if __name__ == "__main__":

    unittest.main()
