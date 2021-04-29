import os
import torch
import numpy as np
from tqdm import trange

from brainspy.algorithms.modules.signal import pearsons_correlation
from brainspy.utils.pytorch import TorchUtils
from brainspy.utils.waveform import process_data


def train(
    model,
    dataloaders,
    criterion,
    optimizer,
    configs,
    logger=None,
    save_dir=None,
    waveform_transforms=None,
    return_best_model=True,
):

    # Evolution loop
    looper = trange(configs["epochs"], desc="Initialising", leave=False)
    pool = optimizer.pool
    best_fitness = -np.inf
    best_correlation = -np.inf
    best_result_index = -1
    genome_history = []
    performance_history = []
    correlation_history = []
    clipping_value = model.get_clipping_value()
    with torch.no_grad():
        model.eval()
        for epoch in looper:
            inputs, targets = dataloaders[0].dataset[:]
            inputs, targets = process_data(waveform_transforms, inputs, targets)
            outputs, criterion_pool = evaluate_population(
                inputs, targets, pool, model, criterion, clipvalue=clipping_value
            )

            # log results
            current_best_index = torch.argmax(
                criterion_pool
            )  # Best output index ignoring nan values

            best_current_output = outputs[current_best_index]
            performance_history.append(
                criterion_pool[current_best_index].detach().cpu()
            )

            genome_history.append(pool[current_best_index].detach().cpu())
            correlation_history.append(
                pearsons_correlation(best_current_output, targets).detach().cpu()
            )
            looper.set_description(
                "  Gen: " + str(epoch + 1) + ". Max fitness: " + str(performance_history[-1].item()) + ". Corr: " + str(correlation_history[-1].item())
            )
            if performance_history[-1] > best_fitness:
                best_fitness = performance_history[-1]
                best_result_index = epoch
                best_correlation = correlation_history[-1].detach().cpu()
                best_output = best_current_output.detach().cpu()
                model.set_control_voltages(genome_history[best_result_index])
                if save_dir is not None:
                    if model.is_hardware():
                        torch.save(
                            model.state_dict(), os.path.join(save_dir, "model.pt")
                        )
                    else:
                        torch.save(model, os.path.join(save_dir, "model.pt"))

            # Check if the best correlation has reached the desired threshold
            if best_correlation >= configs["stop_threshold"]:
                looper.set_description(
                    f"  STOPPED: Correlation {best_correlation} > {configs['stop_threshold']} stopping threshold. "
                )
                looper.close()
                # Close the model adequately if it is on hardware
                if model.is_hardware() and "close" in dir(model):
                    model.close()
                break

            pool = optimizer.step(criterion_pool)

        if return_best_model:  # Return the best model
            if model.is_hardware():
                model.load_state_dict(torch.load(os.path.join(save_dir, "model.pt")))
            else:
                model = torch.load(os.path.join(save_dir, "model.pt"))

        print("Best fitness: " + str(best_fitness.item()))
        return model, {
            "best_result_index": best_result_index,
            "genome_history": genome_history,
            "performance_history": [
                TorchUtils.format(performance_history),
                TorchUtils.format([]),
            ],
            "correlation_history": correlation_history,
            "best_output": best_output,
        }


def evaluate_population(
    inputs, targets, pool, model, criterion, clipvalue=[-np.inf, np.inf]
):
    """Optimisation function of the platform """
    outputs_pool = torch.zeros(
        (len(pool),) + (len(inputs), 1),
        dtype=torch.get_default_dtype(),
        device=TorchUtils.get_device(),
    )
    criterion_pool = torch.zeros(
        len(pool), dtype=torch.get_default_dtype(), device=TorchUtils.get_device()
    )
    for j in range(len(pool)):

        # control_voltage_genes = self.get_control_voltages(gene_pool[j], len(inputs_wfm))  # , gene_pool[j, self.gene_trafo_index]
        # inputs_without_offset_and_scale = self._input_trafo(inputs_wfm, gene_pool[j, self.gene_trafo_index])
        # assert False, 'Check the case for inputing voltages with plateaus to check if it works when merging control voltages and inputs'
        model.set_control_voltages(pool[j])
        outputs_pool[j] = model(inputs)

        if (
            torch.any(outputs_pool[j] <= model.get_clipping_value()[0]) or torch.any(outputs_pool[j] >= model.get_clipping_value()[1]) or (outputs_pool[j] - outputs_pool[j].mean() == 0.0).all()
        ):
            criterion_pool[j] = criterion(None, None, default_value=True)
        else:
            criterion_pool[j] = criterion(outputs_pool[j], targets)

        # output_popul[j] = self.processor.get_output(merge_inputs_and_control_voltages_in_numpy(inputs_without_offset_and_scale, control_voltage_genes, self.input_indices, self.control_voltage_indices))
    return outputs_pool, criterion_pool
