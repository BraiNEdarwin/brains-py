import os
import torch
import random  # TODO: set the seed with TorchUtils or substitute by numpy functions
import warnings
import numpy as np

from tqdm import trange
from typing import Union
from torch.utils.data import DataLoader
from brainspy.utils.pytorch import TorchUtils
from brainspy.utils.signal import pearsons_correlation
from torch.distributions.uniform import Uniform as uniform


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


def train(model: torch.nn.Module,
          dataloaders: list,
          criterion,
          optimizer,
          configs: dict,
          save_dir: str = None,
          average_plateaus: bool = False):
    """
    Main training loop for the genetic algorithm. It supports training a single DNPU hardware
    device on both on and off chip flavours. It only supports using a training dataset (not a
    validation one).

    More information on what a genetic algorithm is can be found at:
    https://towardsdatascience.com/introduction-to-genetic-algorithms-including-example-code-e396e98d8bf3

    Parameters
    ----------
    model : torch.nn.Module
        Model to be trained.

        - The model cannot be an instance of SurrogateModel, HardwareProcessor or Processor.
        - The model can only consist of 1 DNPU instance.
        - The model should have the following methods implemented :

        1. set_control_voltages : To change the control voltages/bias of the network.

                                    Parameters
                                    ----------
                                    bias : torch.Tensor
                                        New value of the bias/control voltage.
                                        One dimensional tensor.

        2. format_targets : The hardware processor uses a waveform to represent points
                            (see 5.1 in Introduction of the Wiki). Each point is represented with some
                            slope and some plateau points. When passing through the hardware, there will
                            be a difference between the output from the device and the input (in points).
                            This function is used for the targets to have the same length in shape as the
                            outputs. It simply repeats each point in the input as many times as there are
                            points in the plateau. In this way, targets can then be compared against hardware
                            outputs in the loss function.

                                    Parameters
                                    ----------
                                    x : torch.Tensor
                                    Targets of the supervised learning problem, that will be extended to have the same
                                    length shape as the outputs from the processor.


        3. is_hardware : Return True if the device is connected to hardware and
                         False if a simulation device is being used

                                    Parameters : None

    dataloaders : list
                  A list containing a single PyTorch Dataloader containing the training dataset.
                  More information about dataloaders can be found at:
                  https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
    criterion : <method>
                Fitness function that will be used to train the model.
    optimizer : GeneticOptimizer
                Optimization method for sorting the genome pool by fitness and creating new
                offspring based on the best resulting genomes.
    configs : dict
        A dictionary containing extra configurations for the algorithm.
            * stop_threshold: float
                When the criterion fitness function reaches the specified threshold, or a higher
                value, the algorithm will stop.
    save_dir : Optional[str]
        Folder where the trained model is going to be saved. When None, the model will not be saved.
        By default None.

    Returns
    -------
    model : torch.nn.Module
        Trained model with best results according to the criterion fitness function.

    training_data: dict
        Dictionary returning relevant data produced while training the model.


    Saved Data
    ----------
    A) After the end of the last epoch, the algorithm saves two main files:
        model_raw.pt: This file is only saved when the model is not hardware (simulation).
        The file is an exact copy of the model after the end of the training process.
        It can be loaded directly as an instance of the model using:
            my_model_instance_at_best_val_results = torch.load('best_model_raw.pt').
        training_data.pickle: A pytorch picle which contains the following keys:
            - epochs: int
                Number of epochs used for training the model
            - algorithm:
                Algorithm type that was being used. Either 'genetic' or 'gradient'.
            - model_state_dict: OrderedDict
                It contains the value of the learnable parameters (weights, or in this case, control voltages) at
                the point where all the training was finised.
            - performance: list
                A list of the fitness function performance over all epochs
            - correlations: list
                A list of the correlation over all epochs
            - genome_history: list
                A list of the genomes that were used in each epoch

    B) If the fitness performance is better than in previous epochs, the following files are saved:
        best_model_raw.pt: This file is only saved when the model is not hardware (simulation). The file is an exact
        copy of the model when it got the best validation results. It can be loaded directly as an instance of the
        model using:
            my_model_instance_at_best_val_results = torch.load('best_model_raw.pt').
        best_training_data.pickle: A pytorch picle which contains the following keys:
            - epoch: int
                Epoch at which the model with best validation loss was found.
            - algorithm: str
                Algorithm type that was being used. Either 'genetic' or 'gradient'.
            - model_state_dict: OrderedDict
                It contains the value of the learnable parameters (weights, or in this case, control voltages)
                at the point where the best validation was achieved.
            - best_fitness: float
                Training fitness at the point where the best validation was achieved.
            - correlation: float
                Correlation achieved on the best fitness achieved.
    """
    assert isinstance(
        model,
        torch.nn.Module), "The model should be an instance of torch.nn.Module"
    assert "set_control_voltages" in dir(
        model
    ), "The set_control_voltages function should be implemeted in the model"
    assert "is_hardware" in dir(
        model), "The is_hardware function should be implemeted in the model"
    assert ("format_targets" in dir(
            model) and not average_plateaus) or average_plateaus, "The format_targets function should be implemeted in the model"
    assert type(dataloaders) == list or type(
        dataloaders) == tuple, "The dataloaders should be of type - list"
    assert len(dataloaders) >= 1 and len(
        dataloaders
    ) < 3, "The dataloaders list should contain a single PyTorch Dataloader"
    assert isinstance(
        dataloaders[0], DataLoader
    ), "The dataloader should be an instance of torch.utils.data.DataLoader"
    assert callable(criterion), "The criterion should be a callable method"
    if not isinstance(optimizer, GeneticOptimizer):
        warnings.warn(
            "The GeneticOptimizer is the only optimizer officially supported. In principle you could use your own custom optimizer with a similar structure to that of GeneticOptimizer. Double check if you really wanted to input an instance of another optimizer than GeneticOptimizer."
        )
    assert type(configs) == dict, "The extra configs should be of type - dict"
    if configs["epochs"]:
        assert type(
            configs["epochs"]) == int, "The epochs key should be of type - int"
    if configs["stop_threshold"]:
        assert type(configs["stop_threshold"]) == float or type(
            configs["stop_threshold"]
        ) == int, "The stop_threshhold key should be of type float or int"
    assert save_dir is None or type(
        save_dir
    ) == str, "The name/path of the save_dir should be of type - str"

    # Evolution loop
    looper = trange(optimizer.epochs, desc="Initialising", leave=False)
    pool = optimizer.pool
    best_fitness = TorchUtils.format([-np.inf], device=torch.device('cpu'))
    best_correlation = TorchUtils.format([-np.inf], device=torch.device('cpu'))
    best_result_index = -1
    genome_history = []
    performance_history = []
    correlation_history = []

    with torch.no_grad():
        model.eval()
        for epoch in looper:
            inputs, targets = dataloaders[0].dataset[:]
            inputs, targets = TorchUtils.format(inputs), TorchUtils.format(
                targets)
            if not average_plateaus:
                targets = model.format_targets(targets)
            outputs, criterion_pool = evaluate_population(
                inputs, targets, pool, model, criterion)


            # log results
            current_best_index = torch.argmax(
                criterion_pool)  # Best output index ignoring nan values

            best_current_output = outputs[current_best_index]
            performance_history.append(
                criterion_pool[current_best_index].detach().cpu())

            genome_history.append(pool[current_best_index].detach().cpu())
            correlation_history.append(
                pearsons_correlation(
                    best_current_output, targets).detach().cpu())  # type: ignore[operator]
            looper.set_description("  Gen: " + str(epoch + 1) +
                                   ". Max fitness: " +
                                   str(performance_history[-1].item()) +
                                   ". Corr: " +
                                   str(correlation_history[-1].item()))
            if performance_history[-1] > best_fitness:
                best_fitness = performance_history[-1]
                best_result_index = epoch
                best_correlation = correlation_history[-1].detach().cpu()
                best_output = best_current_output.detach().cpu()
                model.set_control_voltages(  # type: ignore[operator]
                    genome_history[best_result_index].unsqueeze(0))
                # Only one device is supported,
                # therefore it is unesqueezed for the first dimension.
                if save_dir is not None:
                    torch.save(
                        {
                            "epoch": epoch,
                            "algorithm": 'genetic',
                            "model_state_dict": model.state_dict(),
                            "best_fitness": best_fitness,
                            "correlation": best_correlation
                        },
                        os.path.join(save_dir, "best_training_data.pickle"),
                    )
                    if not model.is_hardware():  # type: ignore[operator]
                        torch.save(model,
                                   os.path.join(save_dir, "best_model_raw.pt"))

                # Check if the correlation of the solution with
                # the best fitness has reached the desired threshold
                if best_correlation >= configs["stop_threshold"]:
                    looper.set_description(
                        f"  STOPPED: Correlation {best_correlation} > {configs['stop_threshold']}"
                        + " stopping threshold. ")
                    looper.close()
                    # Close the model adequately if it is on hardware
                    if model.is_hardware(
                    ) and "close" in dir(  # type: ignore[operator]
                            model):  # type: ignore[operator]
                        model.close()  # type: ignore[operator]
                    break

            pool = optimizer.step(criterion_pool)
        if save_dir is not None:
            torch.save(
                {
                    "epoch": epoch,
                    "algorithm": 'genetic',
                    "model_state_dict": model.state_dict(),
                    "performance": performance_history,
                    "correlations": correlation_history,
                    "genome_history": genome_history
                },
                os.path.join(
                    save_dir,  # type: ignore[arg-type]
                    "training_data.pickle"),
            )
            if not model.is_hardware():  # type: ignore[operator]
                torch.save(
                    model,
                    os.path.join(
                        save_dir,  # type: ignore[arg-type]
                        "model_raw.pt"))  # type: ignore[operator]

            # Load best solution
            model.load_state_dict(
                torch.load(
                    os.path.join(
                        save_dir,  # type: ignore[arg-type]
                        "best_training_data.pickle"))['model_state_dict'])
        print("Best solution in epoch (starting from 0): " +
              str(best_result_index))
        print("Best fitness: " + str(best_fitness.item()))
        print("Correlation: " +
              str(correlation_history[best_result_index].item()))
        return model, {
            "best_result_index":
            best_result_index,
            "genome_history":
            genome_history,
            "performance_history": [
                TorchUtils.format(performance_history),
                TorchUtils.format([]),
            ],
            "correlation_history":
            correlation_history,
            "best_output":
            best_output,
        }


def evaluate_population(inputs: torch.Tensor, targets: torch.Tensor,
                        pool: torch.Tensor, model: torch.nn.Module, criterion):
    """
    Given a particular genome pool, containing all possible control voltage solutions of a genetic
    algorithm, it evaluates on the DNPU model/hardware the fitness for those solutions.

    Parameters
    ----------
    inputs : torch.Tensor
        The whole dataset of inputs in a single batch.
    targets : torch.Tensor
        The whole dataset of target values in a single batch.
    pool : torch.Tensor
        Array of different control voltage values that are going to be evaluated. The array has a
        shape of (pool_size, control_electrode_no).
    model : torch.nn.Module
        Model against which all the solutions will be measured. It can be a Processor, representing
        either a hardware DNPU or a DNPU surrogate model. Refer to the documentation of the train
        function (above) to see how a model can be defined.
    criterion : <method>
                Fitness function that will be used to train the model.

    Returns
    -------
    outputs_pool : torch.Tensor
        All the outputs from all the measurements of the models.

    criterion_pool : torch.Tensor
        Scores of the particular criterion fitness function used in the algorithm. These can be
        used to order the solutions with higher scores.
    """
    assert type(
        inputs) == torch.Tensor, "The inputs should be of type torch.Tensor"
    assert type(
        targets
    ) == torch.Tensor, "The target values should be of type torch.Tensor"
    assert type(
        pool) == torch.Tensor, "The pool should be of type torch.Tensor"
    assert isinstance(
        model,
        torch.nn.Module), "The model should be an instance of torch.nn.Module"
    assert "set_control_voltages" in dir(
        model
    ), "The set_control_voltages function should be implemeted in the model"
    assert "is_hardware" in dir(
        model), "The is_hardware function should be implemeted in the model"
    assert callable(criterion), "The criterion should be a callable method"
    outputs_pool = torch.zeros(
        (len(pool), ) +
        (len(targets), 1),  # type: ignore[operator]
        dtype=torch.get_default_dtype(),
        device=TorchUtils.get_device(),
    )
    criterion_pool = torch.zeros(len(pool),
                                 dtype=torch.get_default_dtype(),
                                 device=TorchUtils.get_device())
    for j in range(len(pool)):

        # control_voltage_genes = self.get_control_voltages(gene_pool[j],
        #  len(inputs_wfm)) #, gene_pool[j, self.gene_trafo_index]
        # inputs_without_offset_and_scale = self._input_trafo(inputs_wfm,
        # gene_pool[j, self.gene_trafo_index])
        # assert False, 'Check the case for inputing voltages with plateaus to check if
        # it works when merging control voltages and inputs'
        model.set_control_voltages(
            pool[j].unsqueeze(0))  # type: ignore[operator]
        outputs_pool[j] = model(inputs)

        if (torch.any(outputs_pool[j] <=
                      model.get_clipping_value()[0])  # type: ignore[operator]
                or torch.any(
                    outputs_pool[j] >=
                    model.get_clipping_value()[1])  # type: ignore[operator]
                or (outputs_pool[j] - outputs_pool[j].mean()
                    == 0.0).all()):  # type: ignore[operator]
            criterion_pool[j] = criterion(
                outputs_pool[j],
                targets,  # type: ignore[operator]
                default_value=True)
        else:
            criterion_pool[j] = criterion(
                outputs_pool[j],
                targets)  # type: ignore[operator]

        # output_popul[j] = self.processor.get_output(merge_inputs_and_control_voltages_in_numpy(
        # inputs_without_offset_ and_scale, control_voltage_genes, self.input_indices,
        # self.control_voltage_indices))
    if (criterion_pool == -1.).all() is True:
        clipping_value = model.get_clipping_value()  # type: ignore[operator]
        warnings.warn(
            "All criterion values are set to -1. This is caused because all the outputs"
            +
            f"are exceeding the clipping values {clipping_value}. This is not a normal "
            +
            "behaviour, you are advised to check the default clipping values in the "
            + "configs, and/or the setup.")
    return outputs_pool, criterion_pool
