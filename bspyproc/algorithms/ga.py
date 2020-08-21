import os
import torch
import numpy as np
from tqdm import trange

from bspyalgo.algorithms.criterion import corr_coeff
from bspyproc.utils.pytorch import TorchUtils


def train(model, dataloaders, criterion, optimizer, configs, logger=None, save_dir=None, waveform_transforms=None, return_best_model=True):

    # Evolution loop
    looper = trange(configs['epochs'], desc='Initialising', leave=False)
    pool = optimizer.pool
    best_fitness = -np.inf
    best_correlation = -np.inf
    best_result_index = -1
    genome_history = []
    performance_history = []
    correlation_history = []
    # outputs_history = []

    for epoch in looper:
        inputs, targets = dataloaders[0].dataset[:]
        outputs = evaluate_population(inputs, pool, model)
        fitness = evaluate_criterion(outputs, targets, criterion, clipvalue=model.get_clipping_value())

        # log results
        no_nan_mask = fitness == fitness
        current_best_index = torch.argmax(fitness[no_nan_mask])  # Best output index ignoring nan values

        best_current_output = outputs[no_nan_mask][current_best_index]
        performance_history.append(fitness[no_nan_mask][current_best_index].detach().cpu())

        genome_history.append(optimizer.pool[no_nan_mask][current_best_index].detach().cpu())
        correlation_history.append(corr_coeff(best_current_output.T, targets.T).detach().cpu())
        looper.set_description("  Gen: " + str(epoch + 1) + ". Max fitness: " + str(performance_history[-1].item()) + ". Corr: " + str(correlation_history[-1].item()))
        if performance_history[-1] > best_fitness:
            best_fitness = performance_history[-1]
            best_result_index = epoch
            best_correlation = correlation_history[-1].detach().cpu()
            best_output = best_current_output.detach().cpu()
            model.set_control_voltages(genome_history[best_result_index])
            if save_dir is not None:
                torch.save(model, os.path.join(save_dir, 'model.pt'))
        # looper.set_description("  Gen: " + str(epoch + 1) + status)
        # self.results['output_current_array'][gen, :, :] = current_state['outputs']
        # self.results['correlation'] = corr_coeff(self.results['best_output'][self.results['mask']].T, self.results['targets'][self.results['mask']].T)
        # # self.data.update({'generation': epoch, 'genes': self.pool, 'outputs': self.outputs, 'fitness': self.fitness})
        # looper.set_description(self.data.get_description(epoch))  # , end - start))

        if best_correlation >= configs['stop_threshold']:
            looper.set_description(f"  STOPPED: Correlation {best_correlation} > {configs['stop_threshold']} stopping threshold. ")
            break

        # if (self.use_checkpoints is True and epoch % self.checkpoint_frequency == 0):
        #    save(mode='pickle', file_path=os.path.join(self.default_checkpoints_dir, 'result.pickle'), data=self.data.results)

        pool = optimizer.step(fitness)

    model.close()
    return model, {'best_result_index': best_result_index, 'genome_history': genome_history, 'performance_history': performance_history, 'correlation_history': correlation_history, 'best_output': best_output}


def evaluate_population(inputs, pool, model):
    '''Optimisation function of the platform '''
    output_popul = TorchUtils.format_tensor(torch.zeros((len(pool),) + (len(inputs), 1)))
    for j in range(len(pool)):

        # control_voltage_genes = self.get_control_voltages(gene_pool[j], len(inputs_wfm))  # , gene_pool[j, self.gene_trafo_index]
        # inputs_without_offset_and_scale = self._input_trafo(inputs_wfm, gene_pool[j, self.gene_trafo_index])
        # assert False, 'Check the case for inputing voltages with plateaus to check if it works when merging control voltages and inputs'
        model.set_control_voltages(pool[j])
        output_popul[j] = model(inputs)
        # output_popul[j] = self.processor.get_output(merge_inputs_and_control_voltages_in_numpy(inputs_without_offset_and_scale, control_voltage_genes, self.input_indices, self.control_voltage_indices))
    return output_popul


def evaluate_criterion(outputs_pool, target, criterion, clipvalue=[-np.inf, np.inf]):
    genome_no = len(outputs_pool)
    criterion_pool = TorchUtils.format_tensor(torch.zeros(genome_no))
    for j in range(genome_no):
        output = outputs_pool[j]
        if torch.any(output < clipvalue[0]) or torch.any(output > clipvalue[1]):
            # print(f'Clipped at {clipvalue} nA')
            result = criterion(None, None, default_value=True)
        else:
            result = criterion(output, target)
        criterion_pool[j] = result
    return criterion_pool
