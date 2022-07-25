import random  # TODO: set the seed with TorchUtils or substitute by numpy functions
import torch
import numpy as np
from typing import Union
from torch.distributions.uniform import Uniform as uniform

from brainspy.utils.pytorch import TorchUtils


class GeneticOptimizer:
    """
    A class for implementing a genetic algorithm optimisation solution for training DNPUs on and
    off chip, in a way that resembles a PyTorch optimizer.
    """
    def __init__(self,
                 gene_ranges: list,
                 partition: Union[list, torch.Tensor],
                 epochs: int,
                 alpha: float = 0.6,
                 beta: float = 0.4):
        """
        Initialises the pool of solutions and the parameters of the optimizer.

        Parameters
        ----------
        gene_ranges : list
            The ranges of the learnable parameters that will be optimised using the algorithm.
        partition : Union[list, torch.Tensor]
            All possible solutions are stored in a pool that is ordered by fitness performance.
            The partition will identify the number of best performing solutions that will not
            mutate on each generation. Expected to be a list or torch.Tensor of length two.
            The first part of the list represents the number of higher fitness genes (best
            performing control voltage solutions) that will remain the same during the mutation.
            The second part of the list specifies the number of genes that will change during
            the mutation step.
        epochs : int
            Number of generations that will be used to mutate the genomes.
        alpha : float, optional
            Alpha parameter for the blend crossover BLX-a-b, by default 0.6. More information on
            the method crossover_blxab.
        beta : float, optional
            Beta parameter for the blend crossover BLX-a-b, by default 0.4. More information on
            the method crossover_blxab.
        """
        assert (type(partition) == torch.Tensor or type(partition) == list)
        assert (type(epochs) == int)
        assert (type(alpha) == float or type(alpha) == int)
        assert (type(beta) == float or type(beta) == int)
        self.epoch = 0
        self.epochs = epochs  # Number of generations
        if isinstance(gene_ranges, list):
            self.gene_range = TorchUtils.format(np.array(gene_ranges))
        else:
            self.gene_range = gene_ranges
        assert not torch.eq(self.gene_range[:, 0], self.gene_range[:, 1]).any(
        ), "Min and Max control voltage ranges should be different from each other."
        assert (self.gene_range[:, 0] < self.gene_range[:, 1]).all(
        ), "Min control voltage ranges should be lower than max control voltage ranges."
        self.partition = partition
        self.genome_no = sum(self.partition)
        self.pool = self._init_pool()
        # Parameters for crossover
        self.alpha = alpha
        self.beta = beta

    def step(self, criterion_pool: torch.Tensor):
        """
        This function performs an epoch step for a new generation of solutions. First, it sorts
        the gene pool based on fitness performance. Then, it generates offspring between best
        solutions by applying the blend crossover alpha-beta (BLX-a-b). Finally, it mutates
        every genome except those specified not to be updated in the partiton variable.

        Parameters
        ----------
        criterion_pool : torch.Tensor
            A pool storing the results from the criterion (fitness function) in the same order as
            that of the genome pool containing all solutions. It is used to help sorting the pool
            solutions by fitness.

        Returns
        -------
        pool
            Genome pool containing the set of all control voltage solutions ordered by fitness
            performance.
        """
        assert (type(criterion_pool) == torch.Tensor)
        # Sort gene pool based on fitness and copy it
        self.pool = self.pool[torch.flip(
            torch.argsort(criterion_pool), dims=[0]
        )]  # WARNING: old code was np.argsort(fitness)[::-1], changed it as it is because it was
        # giving an error. I assume that fitness will always have one dimension,
        # with the number of genomes.

        # Generate offspring by means of crossover.
        # The crossover method returns 1 genome from 2 parents.
        new_pool = self.crossover(self.pool.clone(), self.universal_sampling())

        # Mutate every genome except the partition[0]
        self.pool = self.mutation(new_pool).clone()

        # Free memory space and count epoch (generation)
        del new_pool
        self.epoch += 1
        return self.pool

    def _init_pool(self):
        """
        Initialises a set of possible control voltage solutions using random values that fall
        within a specific range.

        Returns
        -------
        pool : torch.Tensor
            A pool containing random control voltage solutions that fall within a specific range.
        """
        pool = torch.zeros(
            (self.genome_no, len(self.gene_range)),
            device=TorchUtils.get_device(),
            dtype=torch.get_default_dtype(),
        )  # Dimensions (Genome number, gene number)
        for i in range(0, len(self.gene_range)):
            pool[:, i] = uniform(self.gene_range[i][0],
                                 self.gene_range[i][1]).sample(
                                     (self.genome_no, ))
        return pool

    def crossover(self, new_pool: torch.Tensor, chosen: list):
        """
        In genetic algorithms and evolutionary computation, crossover, also called recombination,
        is a genetic operator used to combine the genetic information of two parents to generate
        new offspring. It is one way to stochastically generate new solutions from an existing
        population, and is analogous to the crossover that happens during reproduction in
        biology.

        Parameters
        ----------
        new_pool : torch.Tensor
            A copy of the genome pool containing the set of all control voltage solutions ordered
            by fitness performance.
        chosen: list
            Parents that have been chosen for potentially good solutions.

        Returns
        -------
        new_pool : torch.Tensor
            A genome pool containing the set of all control voltage solutions obtained by
            applying the crossover method against those solutions with higest fitness
            scores.
        """
        # length = self.len(fitness)

        # Determine which genomes are chosen to generate offspring
        # Note: twice as much parents are selected as there are genomes to be generated
        assert (type(new_pool) == torch.Tensor)
        assert (type(chosen) == list)
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
                    self.pool[chosen[i], :], self.pool[chosen[i + 1], :])
            else:
                new_pool[index_newpool, :] = self.crossover_blxab(
                    self.pool[chosen[i + 1], :], self.pool[chosen[i], :])
        return new_pool

    def universal_sampling(self):
        """
        A technique used in genetic algorithms for selecting potentially useful solutions for
        crossover.
        More information can be found in:
        https://en.wikipedia.org/wiki/Stochastic_universal_sampling#cite_note-baker-1


        Returns
        -------
        list
            The chosen 'parents'. length: len(self.fitness) == len(self.pool).
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
                elif pointer < probabilities[i] and pointer >= probabilities[
                        i - 1]:
                    chosen.append(i)
                    break
        chosen = random.sample(chosen, len(chosen))
        return chosen

    def linear_rank(self):
        """
        Linear ranking scheme used for stochastic universal sampling method.

        Returns
        -------
        torch.Tensor

        Tensor with the probability of a genome being chosen. The first probability corresponds to
        the genome with the highest fitness, etc.
        """
        maximum = 1.5
        rank = np.arange(self.genome_no) + 1
        minimum = 2 - maximum
        probability = (minimum + ((maximum - minimum) * (rank - 1) /
                                  (self.genome_no - 1))) / self.genome_no
        return probability[::-1]

    def crossover_blxab(self, parent1: torch.Tensor, parent2: torch.Tensor):
        """
        Creates a new offspring by selecting a random value from the interval between the two
        alleles of the parent solutions. The interval is increased in direction of the solution
        with better fitness by the factor alpha, and into the direction of the solution with worse
        fitness by the factor beta.
        Crossover method: Blend alpha beta crossover returns a new genome (voltage combination)
        from two parents. Here, parent 1 has a higher fitness than parent 2

        Parameters
        ----------
        parent1 : torch.Tensor
            Set of control voltages corresponding to a particular solution that has higher or equal
            fitness than parent2.
        parent2 : torch.Tensor
            Set of control voltages corresponding to a particular solution that has lower or equal
            fitness than parent1.

        Returns
        -------
        torch.Tensor
            New genome (voltage combination) from two parent control voltages.
        """
        # check this in pytorch
        assert (type(parent1) == torch.Tensor and type(parent2)
                == torch.Tensor), "Parents are not torch Tensors."
        assert (parent1.device == parent2.device
                ), "Parent Tensors are not in the same device."
        assert (parent1.dtype == parent2.dtype
                ), "Parent Tensors are not of the same datatype."
        maximum = torch.max(parent1, parent2)
        minimum = torch.min(parent1, parent2)
        diff_maxmin = maximum - minimum
        offspring = torch.zeros(
            (parent1.shape),
            dtype=parent1.dtype,
            device=parent1.device,
        )
        for i in range(len(parent1)):
            if parent1[i] > parent2[i]:
                offspring[i] = uniform(
                    minimum[i] - diff_maxmin[i] * self.beta,
                    maximum[i] + diff_maxmin[i] * self.alpha,
                ).sample()
            elif parent1[i] == parent2[i]:
                offspring[i] = parent1[i].clone()
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
        Dynamic parameter control of mutation rate. This method updates the mutation
        rate based on the generation counter.

        Returns
        -------
        float
            Mutation rate parameter.
        """

        pm_inv = 2 + 5 / (self.epochs - 1) * self.epoch
        assert pm_inv > 0
        return 0.625 / pm_inv

    def mutation(self, pool: torch.Tensor):
        """
        Mutate all genes but the first partition[0], with a triangular
        distribution in gene range with mode=gene to be mutated.

        Parameters
        ----------
        pool : torch.Tensor
            Genome pool containing the set of all control voltage solutions ordered by fitness
            performance.

        Returns
        -------
        pool : torch.Tensor
            Genome pool containing a new mutated set of all control voltage solutions based on the
            best performing solutions.
        """
        assert (type(pool) == torch.Tensor)
        # The mutation rate is updated based on the generation counter
        mutation_rate = self.update_mutation_rate()

        # Check if the mask requires to
        mask = TorchUtils.format(
            np.random.choice(
                [0, 1],
                size=pool[self.partition[0]:].shape,  # type: ignore[misc]
                p=[1 - mutation_rate, mutation_rate],
            ))
        mutated_pool = np.zeros(
            (self.genome_no - self.partition[0], len(self.gene_range)))
        gene_range = TorchUtils.to_numpy(self.gene_range)
        for i in range(0, len(gene_range)):
            mutated_pool[:, i] = np.random.triangular(
                gene_range[i][0],
                TorchUtils.to_numpy(
                    pool[self.partition[0]:,  # type: ignore[misc]
                         i]),  # type: ignore[misc]
                gene_range[i][1],
            )

        mutated_pool = TorchUtils.format(mutated_pool)
        pool[self.partition[0]:] = (  # type: ignore[misc]
            torch.ones(
                pool[self.partition[0]:].shape,  # type: ignore[misc]
                dtype=torch.get_default_dtype(),
                device=TorchUtils.get_device(),
            ) - mask) * pool[
                self.partition[0]:] + mask * mutated_pool  # type: ignore[misc]

        # Remove duplicates (Only if they are)
        if len(pool.unique(dim=0)) < len(pool):
            pool = self.remove_duplicates(pool)

        return pool

    def remove_duplicates(self, pool: torch.Tensor):
        """
        Check the entire pool for any duplicate genomes and replace them by
        the genome put through a triangular distribution.

        Parameters
        ----------
        pool : torch.Tensor
            Genome pool containing a new mutated set of all control voltage solutions based on the
            best performing solutions.

        Returns
        -------
        pool : torch.Tensor
            Genome pool containing a new mutated set of all control voltage solutions based on the
            best performing solutions without any repeated solution.
        """
        assert (type(pool) == torch.Tensor)
        if torch.unique(pool, dim=0).shape != pool.shape:
            for i in range(self.genome_no):
                for j in range(self.genome_no):
                    if j != i and torch.eq(pool[i], pool[j]).all():
                        for k in range(0, len(self.gene_range)):
                            pool[j][k] = TorchUtils.format(
                                [
                                    np.random.triangular(
                                        self.gene_range[k][0].item(),
                                        pool[j][k].item(),
                                        self.gene_range[k][1].item(),
                                    )
                                ],
                                device=pool.device,
                                data_type=pool.dtype).squeeze()
        return pool
