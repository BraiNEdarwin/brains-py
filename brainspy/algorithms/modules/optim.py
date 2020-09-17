import random  # TODO: set the seed with TorchUtils or substitute by numpy functions
import torch
from torch.distributions.uniform import Uniform as uniform
import numpy as np

from brainspy.utils.pytorch import TorchUtils

# %%
#    ##########################################################################
#    ##################### Methods defining evolution #########################
#    ##########################################################################
# ------------------------------------------------------------------------------
""" Contains class implementing the Genetic Algorithm for all SkyNEt platforms.
Created on Thu May 16 18:16:36 2019
@author: HCRuiz, A. Uitzetter and Unai Alegre
"""


class GeneticOptimizer:
    def __init__(self, gene_ranges, partition, epochs, alpha=0.6, beta=0.4):
        self.epoch = 0
        self.epochs = epochs  # Number of generations
        if isinstance(gene_ranges, list):
            self.gene_range = TorchUtils.get_tensor_from_list(gene_ranges)
        else:
            self.gene_range = gene_ranges
        self.partition = partition
        self.genome_no = sum(self.partition)
        self.pool = self._init_pool()
        # Parameters for crossover
        self.alpha = alpha
        self.beta = beta

    def step(self, criterion_pool):
        # Sort gene pool based on fitness and copy it
        self.pool = self.pool[
            torch.flip(torch.argsort(criterion_pool), dims=[0])
        ]  # WARNING: old code was np.argsort(fitness)[::-1], changed it as it is because it was giving an error. I assume that fitness will always have one dimension, with the number of genomes.

        # Generate offspring by means of crossover. The crossover method returns 1 genome from 2 parents.
        new_pool = self.crossover(self.pool.clone())

        # Mutate every genome except the partition[0]
        self.pool = self.mutation(new_pool).clone()

        # Free memory space and count epoch (generation)
        del new_pool
        self.epoch += 1
        return self.pool

    def _init_pool(self):
        pool = torch.zeros((self.genome_no, len(self.gene_range)), device=TorchUtils.get_accelerator_type(), dtype=TorchUtils.get_data_type())  # Dimensions (Genome number, gene number)
        for i in range(0, len(self.gene_range)):
            pool[:, i] = uniform(self.gene_range[i][0], self.gene_range[i][1]).sample((self.genome_no,))
        return pool

    def crossover(self, new_pool):
        # length = self.len(fitness)

        # Determine which genomes are chosen to generate offspring
        # Note: twice as much parents are selected as there are genomes to be generated
        chosen = self.universal_sampling()
        for i in range(0, len(chosen), 2):
            index_newpool = int(i / 2 + sum(self.partition[:1]))
            if chosen[i] == chosen[i + 1]:
                if chosen[i] == 0:
                    chosen[i] = chosen[i] + 1
                else:
                    chosen[i] = chosen[i] - 1

            # The individual with the highest fitness score is given as input first
            if chosen[i] < chosen[i + 1]:
                new_pool[index_newpool, :] = self.crossover_blxab(
                    self.pool[chosen[i], :], self.pool[chosen[i + 1], :]
                )
            else:
                new_pool[index_newpool, :] = self.crossover_blxab(
                    self.pool[chosen[i + 1], :], self.pool[chosen[i], :]
                )
        return new_pool

    def universal_sampling(self):
        """
        Sampling method: Stochastic universal sampling returns the chosen 'parents'
        length: len(self.fitness) == len(self.pool)
        """
        no_genomes = 2 * self.partition[1]
        chosen = []
        probabilities = self.linear_rank()
        for i in range(1, len(self.pool)):
            probabilities[i] = probabilities[i] + probabilities[i - 1]
        distance = 1 / (no_genomes)
        start = random.random() * distance
        for n in range(no_genomes):
            pointer = start + n * distance
            for i in range(len(self.pool)):
                if pointer < probabilities[0]:
                    chosen.append(0)
                    break
                elif pointer < probabilities[i] and pointer >= probabilities[i - 1]:
                    chosen.append(i)
                    break
        chosen = random.sample(chosen, len(chosen))
        return chosen

    def linear_rank(self):
        """
        Assigning probabilities: Linear ranking scheme used for stochastic universal sampling method.
        It returns an array with the probability that a genome will be chosen.
        The first probability corresponds to the genome with the highest fitness etc.
        """
        maximum = 1.5
        rank = np.arange(self.genome_no) + 1
        minimum = 2 - maximum
        probability = (
            minimum + ((maximum - minimum) * (rank - 1) / (self.genome_no - 1))
        ) / self.genome_no
        return probability[::-1]

    def crossover_blxab(self, parent1, parent2):
        """
        Crossover method: Blend alpha beta crossover returns a new genome (voltage combination)
        from two parents. Here, parent 1 has a higher fitness than parent 2
        """

        # check this in pytorch
        maximum = torch.max(parent1, parent2)
        minimum = torch.min(parent1, parent2)
        diff_maxmin = maximum - minimum
        offspring = torch.zeros((parent1.shape), dtype=TorchUtils.get_data_type(), device=TorchUtils.get_accelerator_type())
        for i in range(len(parent1)):
            if parent1[i] > parent2[i]:
                offspring[i] = uniform(
                    minimum[i] - diff_maxmin[i] * self.beta,
                    maximum[i] + diff_maxmin[i] * self.alpha,
                ).sample()
            else:
                offspring[i] = uniform(
                    minimum[i] - diff_maxmin[i] * self.alpha,
                    maximum[i] + diff_maxmin[i] * self.beta,
                ).sample()
        for i in range(0, len(self.gene_range)):
            if offspring[i] < self.gene_range[i][0]:
                offspring[i] = self.gene_range[i][0]
            if offspring[i] > self.gene_range[i][1]:
                offspring[i] = self.gene_range[i][1]
        return offspring

    def update_mutation_rate(self):
        """
        Dynamic parameter control of mutation rate: This formula updates the mutation
        rate based on the generation counter
        """
        pm_inv = 2 + 5 / (self.epochs - 1) * self.epoch
        assert pm_inv > 0
        return 0.625 / pm_inv

    def mutation(self, pool):
        """
        Mutate all genes but the first partition[0] with a triangular
        distribution in gene range with mode=gene to be mutated.
        """

        # The mutation rate is updated based on the generation counter
        mutation_rate = self.update_mutation_rate()

        # Check if the mask requires to
        mask = TorchUtils.get_tensor_from_numpy(
            np.random.choice(
                [0, 1],
                size=pool[self.partition[0]:].shape,
                p=[1 - mutation_rate, mutation_rate],
            )
        )
        mutated_pool = np.zeros(
            (self.genome_no - self.partition[0], len(self.gene_range))
        )
        gene_range = TorchUtils.get_numpy_from_tensor(self.gene_range)
        for i in range(0, len(gene_range)):
            if gene_range[i][0] == gene_range[i][1]:
                mutated_pool[:, i] = gene_range[i][0] * np.ones(
                    mutated_pool[:, i].shape
                )
            else:
                mutated_pool[:, i] = np.random.triangular(
                    gene_range[i][0],
                    TorchUtils.get_numpy_from_tensor(pool[self.partition[0]:, i]),
                    gene_range[i][1],
                )

        mutated_pool = TorchUtils.get_tensor_from_numpy(mutated_pool)
        pool[self.partition[0]:] = (
            torch.ones(pool[self.partition[0]:].shape, dtype=TorchUtils.get_data_type(), device=TorchUtils.get_accelerator_type()) - mask
        ) * pool[self.partition[0]:] + mask * mutated_pool

        # Remove duplicates (Only if they are)
        if len(pool.unique(dim=1)) < len(pool):
            pool = self.remove_duplicates(pool)

        return pool

    def remove_duplicates(self, pool):
        """
        Check the entire pool for any duplicate genomes and replace them by
        the genome put through a triangular distribution
        """
        if torch.unique(pool, dim=0).shape != pool.shape:
            for i in range(self.genome_no):
                for j in range(self.genome_no):
                    if j != i and torch.eq(pool[i], pool[j]).all():
                        for k in range(0, len(self.gene_range)):
                            if self.gene_range[k][0] != self.gene_range[k][1]:
                                pool[j][k] = TorchUtils.get_tensor_from_numpy(
                                    np.random.triangular(
                                        self.gene_range[k][0],
                                        pool[j][k],
                                        self.gene_range[k][1],
                                    )
                                )
                            else:
                                pool[j][k] = self.gene_range[k][0]
        return pool
