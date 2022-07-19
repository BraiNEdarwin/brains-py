import unittest
import torch
import numpy as np
from brainspy.utils.pytorch import TorchUtils
from brainspy.algorithms.modules.optim import GeneticOptimizer


class OptimTest(unittest.TestCase):
    """
    Testing the optim.py file - GeneticOptimizer class.
    """

    def test_init(self):
        """
        Test to initialize the Genetic Optimizer
        """
        try:
            optim = GeneticOptimizer(gene_ranges=TorchUtils.format(
                torch.tensor([[-1.2, 0.6], [-1.2, 0.6]])),
                                     partition=[4, 22],
                                     epochs=100)

            self.assertEqual(optim.epochs, 100)
            self.assertEqual(optim.epoch, 0)
            self.assertEqual(optim.alpha, 0.6)
            self.assertEqual(optim.beta, 0.4)
            self.assertEqual(
                optim.partition,
                [torch.tensor(4), torch.tensor(22)])
            self.assertEqual(optim.genome_no,
                             sum([torch.tensor(4),
                                  torch.tensor(22)]))
            assert isinstance(optim.gene_range, torch.Tensor)
        except (Exception):
            self.fail("Couldnt initaialize GeneticOptimizer")

    def test_init_max_min(self):
        """
        if max value of gene ranges is less than the min value,
        a ValueError is raised
        """
        with self.assertRaises(ValueError):
            GeneticOptimizer(gene_ranges=torch.tensor([[1.2, 0.6], [1.2,
                                                                    0.6]]),
                             partition=[torch.tensor(4),
                                        torch.tensor(22)],
                             epochs=100)

    def test_init_invalid_shape(self):
        """
        Invalid shape for gene ranges raises an Index Error
        """
        with self.assertRaises(IndexError):
            GeneticOptimizer(gene_ranges=torch.tensor([[1.2], [0.6]]),
                             partition=[torch.tensor(4),
                                        torch.tensor(22)],
                             epochs=100)

    def test_init_invalid_type(self):
        """
        Invalid type for gene ranges raises an IndexError
        """
        with self.assertRaises(IndexError):
            GeneticOptimizer(gene_ranges=[-1.2, 0.6],
                             partition=[torch.tensor(4),
                                        torch.tensor(22)],
                             epochs=100)

    def test_init_invalid_none(self):
        """
        None value for gene ranges raises a TypeError
        """
        with self.assertRaises(TypeError):
            GeneticOptimizer(gene_ranges=None,
                             partition=[torch.tensor(4),
                                        torch.tensor(22)],
                             epochs=100)

    def test_init_invalid_negative_dim(self):
        """
        Negative value for partition raises a RuntimeError
        - Trying to create tensor with negative dimension -11: [-11, 2]
        """
        with self.assertRaises(RuntimeError):
            GeneticOptimizer(gene_ranges=torch.tensor([[-1.2, 0.6],
                                                       [-1.2, 0.6]]),
                             partition=[torch.tensor(-11)],
                             epochs=100)

    def test_init_invalid_type_partition(self):
        """
        Invalid type for partition raises an AssertionError
        """
        with self.assertRaises(AssertionError):
            GeneticOptimizer(gene_ranges=torch.tensor([[-1.2, 0.6],
                                                       [-1.2, 0.6]]),
                             partition=np.array([1, 2, 3, 4]),
                             epochs=100)

        with self.assertRaises(AssertionError):
            GeneticOptimizer(gene_ranges=torch.tensor([[-1.2, 0.6],
                                                       [-1.2, 0.6]]),
                             partition=None,
                             epochs=100)

        with self.assertRaises(AssertionError):
            GeneticOptimizer(gene_ranges=torch.tensor([[-1.2, 0.6],
                                                       [-1.2, 0.6]]),
                             partition="string type",
                             epochs=100)

    def test_init_invalid_type_epochs(self):
        """
        Invalid type for epochs raises an AssertionError
        """
        with self.assertRaises(AssertionError):
            GeneticOptimizer(gene_ranges=torch.tensor([[-1.2, 0.6],
                                                       [-1.2, 0.6]]),
                             partition=[torch.tensor(4),
                                        torch.tensor(22)],
                             epochs=[1, 2, 3, 4])

        with self.assertRaises(AssertionError):
            GeneticOptimizer(gene_ranges=torch.tensor([[-1.2, 0.6],
                                                       [-1.2, 0.6]]),
                             partition=[torch.tensor(4),
                                        torch.tensor(22)],
                             epochs="invalid type")

        with self.assertRaises(AssertionError):
            GeneticOptimizer(gene_ranges=torch.tensor([[-1.2, 0.6],
                                                       [-1.2, 0.6]]),
                             partition=[torch.tensor(4),
                                        torch.tensor(22)],
                             epochs=np.array([1, 2, 3, 4]))

    def test_step(self):
        """
        Testing the step function with a random torch tensor for a
        criterion pool and checking if epoch increments
        """
        optim = GeneticOptimizer(gene_ranges=torch.tensor([[-1.2, 0.6],
                                                           [-1.2, 0.6]]),
                                 partition=[4, 22],
                                 epochs=100)
        try:
            optim.step(criterion_pool=torch.randint(1, 100, (1, 26)).squeeze())
        except (Exception):
            self.fail("Could'nt perform step")
        self.assertEqual(optim.epoch, 1)

    def test_step_val_error(self):
        """
        ValueError: step must be greater than zero
        """
        with self.assertRaises(ValueError):
            optim = GeneticOptimizer(
                gene_ranges=torch.tensor([[-1.2, 0.6], [-1.2, 0.6]]),
                partition=[torch.tensor(4), torch.tensor(22)],
                epochs=100)
            optim.step(criterion_pool=torch.tensor([1, 2, 3, 4]))

        with self.assertRaises(ValueError):
            optim = GeneticOptimizer(
                gene_ranges=torch.tensor([[-1.2, 0.6], [-1.2, 0.6]]),
                partition=[torch.tensor(4), torch.tensor(22)],
                epochs=100)
            optim.step(
                criterion_pool=torch.randint(-100, -1, (1, 26)).squeeze())

    def test_step_fail(self):
        """
        Invalid type for step argument raises an AssertionError
        """
        optim = GeneticOptimizer(gene_ranges=torch.tensor([[-1.2, 0.6],
                                                           [-1.2, 0.6]]),
                                 partition=[4, 22],
                                 epochs=100)
        with self.assertRaises(AssertionError):
            optim.step(None)
        with self.assertRaises(AssertionError):
            optim.step(np.array([1, 2, 3, 4]))
        with self.assertRaises(AssertionError):
            optim.step("string type")
        with self.assertRaises(AssertionError):
            optim.step(1000)

    def test_init_pool(self):
        """
        Test to generate a random pool
        """
        optim = GeneticOptimizer(gene_ranges=torch.tensor([[-1.2, 0.6],
                                                           [-1.2, 0.6]]),
                                 partition=[4, 22],
                                 epochs=100)
        try:
            optim._init_pool()
        except (Exception):
            self.fail("Could not generate random pool")

    def test_crossover(self):
        """
        Testing the crossover method with a random torch tensor of correct dimension
        """
        optim = GeneticOptimizer(gene_ranges=torch.tensor([[-1.2, 0.6],
                                                           [-1.2, 0.6]]),
                                 partition=[4, 22],
                                 epochs=100)

        try:
            criterion_pool = torch.randint(1, 100, (1, 26)).squeeze()
            optim.pool = optim.pool[torch.flip(torch.argsort(criterion_pool),
                                               dims=[0])]
            new_pool = optim.crossover(new_pool=optim.pool)
        except (Exception):
            self.fail("Could not generate random pool")
        else:
            assert isinstance(new_pool, torch.Tensor)

    def test_crossover_fail_IndexError(self):
        """
        IndexError is raised if an incorrect dimension is provided as an input for
        the crossover method
        """
        optim = GeneticOptimizer(gene_ranges=torch.tensor([[-1.2, 0.6],
                                                           [-1.2, 0.6]]),
                                 partition=[4, 22],
                                 epochs=100)

        with self.assertRaises(IndexError):
            criterion_pool = torch.randint(1, 100, (1, 20)).squeeze()
            optim.pool = optim.pool[torch.flip(torch.argsort(criterion_pool),
                                               dims=[0])]
            optim.crossover(new_pool=optim.pool)

    def test_crossover_fail(self):
        """
        Invalid type for the crossover method raises an AssertionError
        """
        optim = GeneticOptimizer(gene_ranges=torch.tensor([[-1.2, 0.6],
                                                           [-1.2, 0.6]]),
                                 partition=[4, 22],
                                 epochs=100)
        with self.assertRaises(AssertionError):
            optim.crossover("String type")
        with self.assertRaises(AssertionError):
            optim.crossover(np.array([1, 2, 3]))
        with self.assertRaises(AssertionError):
            optim.crossover(100)
        with self.assertRaises(AssertionError):
            optim.crossover([1, 2, 3, 4])

    def test_universal_sampling(self):
        """
        Test for Universal sampling after an initial step
        """
        optim = GeneticOptimizer(gene_ranges=torch.tensor([[-1.2, 0.6],
                                                           [-1.2, 0.6]]),
                                 partition=[4, 22],
                                 epochs=100)
        optim.step(criterion_pool=torch.randint(1, 100, (1, 26)).squeeze())
        try:
            optim.universal_sampling()
        except (Exception):
            self.fail("Could not run universal sampler method")

    def test_linear_rank(self):
        """
        Test for linear ranking after an initial step
        """
        optim = GeneticOptimizer(gene_ranges=torch.tensor([[-1.2, 0.6],
                                                           [-1.2, 0.6]]),
                                 partition=[4, 22],
                                 epochs=100)
        optim.step(criterion_pool=torch.randint(1, 100, (1, 26)).squeeze())
        try:
            optim.linear_rank()
        except (Exception):
            self.fail("Could not get linear rank")

    def test_crossover_blxab(self):
        """
        Test for the crossover_blxab() method with 2 random values for
        parents
        """
        optim = GeneticOptimizer(gene_ranges=torch.tensor([[-1.2, 0.6],
                                                           [-1.2, 0.6]]),
                                 partition=[4, 22],
                                 epochs=100)
        try:
            offspring = optim.crossover_blxab(torch.rand(10), torch.rand(10))
            assert isinstance(offspring, torch.Tensor)
        except (Exception):
            self.fail("Couldn do alpha beta crossover ")

    def test_crossover_blxab_fail(self):
        """
        Invalid type for crossover_blxab() raises an AssertionError
        """
        optim = GeneticOptimizer(gene_ranges=torch.tensor([[-1.2, 0.6],
                                                           [-1.2, 0.6]]),
                                 partition=[4, 22],
                                 epochs=100)
        with self.assertRaises(AssertionError):
            optim.crossover_blxab("torch.rand(10)", torch.rand(10))
        with self.assertRaises(AssertionError):
            optim.crossover_blxab(100, torch.rand(10))
        with self.assertRaises(AssertionError):
            optim.crossover_blxab(torch.rand(10), np.array([1, 2, 3, 4]))
        with self.assertRaises(AssertionError):
            optim.crossover_blxab("torch.rand(10)", [1, 2, 3, 4])

    def test_update_mutation_rate(self):
        """
        Test for update_mutation_rate() method
        """
        optim = GeneticOptimizer(gene_ranges=torch.tensor([[-1.2, 0.6],
                                                           [-1.2, 0.6]]),
                                 partition=[4, 22],
                                 epochs=100)
        try:
            optim.update_mutation_rate()
        except (Exception):
            self.fail("Could not update mutataion rate")

    def test_mutation(self):
        """
        Test for a mutation with a pool after an initial step and crossover
        """
        optim = GeneticOptimizer(gene_ranges=torch.tensor([[-1.2, 0.6],
                                                           [-1.2, 0.6]]),
                                 partition=[4, 22],
                                 epochs=100)
        try:
            criterion_pool = torch.randint(1, 100, (1, 26)).squeeze()
            optim.pool = optim.pool[torch.flip(torch.argsort(criterion_pool),
                                               dims=[0])]
            new_pool = optim.crossover(new_pool=optim.pool)
            optim.mutation(new_pool)
        except (Exception):
            self.fail("Could not update mutataion rate")

    def test_mutataion_fail(self):
        """
        Invalid type for muttation raises an AssertionError
        """
        optim = GeneticOptimizer(gene_ranges=torch.tensor([[-1.2, 0.6],
                                                           [-1.2, 0.6]]),
                                 partition=[4, 22],
                                 epochs=100)
        with self.assertRaises(AssertionError):
            optim.mutation("pool")
        with self.assertRaises(AssertionError):
            optim.mutation(np.array([1, 2, 3, 4]))
        with self.assertRaises(AssertionError):
            optim.mutation([1, 2, 3, 4, 5])
        with self.assertRaises(AssertionError):
            optim.mutation(100)

    def test_remove_duplicates(self):
        """
        Test to remove duplicates after a crossover
        """
        optim = GeneticOptimizer(gene_ranges=torch.tensor([[-1.2, 0.6],
                                                           [-1.2, 0.6]]),
                                 partition=[4, 22],
                                 epochs=100)
        try:
            criterion_pool = torch.randint(1, 100, (1, 26)).squeeze()
            optim.pool = optim.pool[torch.flip(torch.argsort(criterion_pool),
                                               dims=[0])]
            new_pool = optim.crossover(new_pool=optim.pool)
            edited = optim.remove_duplicates(new_pool)
        except (Exception):
            self.fail("Could not update mutataion rate")
        assert (len(new_pool) >= len(edited))

    def test_remove_duplicates_fail(self):
        """
        Invalid type for remove_duplicates raises an AssertionError
        """
        optim = GeneticOptimizer(gene_ranges=torch.tensor([[-1.2, 0.6],
                                                           [-1.2, 0.6]]),
                                 partition=[4, 22],
                                 epochs=100)
        with self.assertRaises(AssertionError):
            optim.remove_duplicates("pool")
        with self.assertRaises(AssertionError):
            optim.remove_duplicates(np.array([1, 2, 3, 4]))
        with self.assertRaises(AssertionError):
            optim.remove_duplicates([1, 2, 3, 4, 5])
        with self.assertRaises(AssertionError):
            optim.remove_duplicates(100)


if __name__ == "__main__":
    unittest.main()
