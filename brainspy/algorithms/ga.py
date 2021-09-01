import os
import torch
import numpy as np
from tqdm import trange

from brainspy.algorithms.modules.signal import pearsons_correlation
from brainspy.utils.pytorch import TorchUtils


def train(model: torch.nn.Module,
          dataloaders: list,
          criterion,
          optimizer,
          configs: dict,
          save_dir: str = None):
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
            * epochs : int
                Number of steps (generations) that the algorithm will take in order to train the
                model.
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
    """
    # Evolution loop
    looper = trange(configs["epochs"], desc="Initialising", leave=False)
    pool = optimizer.pool
    best_fitness = -np.inf
    best_correlation = -np.inf
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
                    best_current_output,
                    model.format_targets(targets)).detach().cpu())
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
                model.set_control_voltages(
                    genome_history[best_result_index].unsqueeze(
                        0))  # Only one device is supported,
                # therefore it is unesqueezed for the first dimension.
                if save_dir is not None:
                    torch.save(model.state_dict(),
                               os.path.join(save_dir, "model.pt"))
                    if not model.is_hardware():
                        torch.save(model, os.path.join(save_dir,
                                                       "model_raw.pt"))

            # Check if the best correlation has reached the desired threshold
            if best_correlation >= configs["stop_threshold"]:
                looper.set_description(
                    f"  STOPPED: Correlation {best_correlation} > {configs['stop_threshold']}"
                    + " stopping threshold. ")
                looper.close()
                # Close the model adequately if it is on hardware
                if model.is_hardware() and "close" in dir(model):
                    model.close()
                break

            pool = optimizer.step(criterion_pool)

        print("Best fitness: " + str(best_fitness.item()))
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
        [description]
    model : torch.nn.Module
        Model against which all the solutions will be measured. It can be a Processor, representing
        either a hardware DNPU or a DNPU surrogate model.
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
    outputs_pool = torch.zeros(
        (len(pool), ) + (len(model.format_targets(inputs)), 1),
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
        model.set_control_voltages(pool[j].unsqueeze(0))
        outputs_pool[j] = model(inputs)

        if (torch.any(outputs_pool[j] <= model.get_clipping_value()[0])
                or torch.any(outputs_pool[j] >= model.get_clipping_value()[1])
                or (outputs_pool[j] - outputs_pool[j].mean() == 0.0).all()):
            criterion_pool[j] = criterion(outputs_pool[j],
                                          model.format_targets(targets),
                                          default_value=True)
        else:
            criterion_pool[j] = criterion(outputs_pool[j],
                                          model.format_targets(targets))

        # output_popul[j] = self.processor.get_output(merge_inputs_and_control_voltages_in_numpy(
        # inputs_without_offset_ and_scale, control_voltage_genes, self.input_indices,
        # self.control_voltage_indices))
    return outputs_pool, criterion_pool
