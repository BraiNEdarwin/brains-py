import torch
from tqdm import trange
import numpy as np
import os

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
    # with torch.autograd.profiler.profile(use_cuda=not TorchUtils.force_cpu) as prof:
    train_losses, val_losses = [], []
    min_val_loss = np.inf
    looper = trange(configs["epochs"], desc=" Initialising")
    model.to(device=TorchUtils.get_accelerator_type())

    if 'set_regul_factor' in dir(model) and 'regul_factor' in configs:
        model.set_regul_factor(configs['regul_factor'])

    for epoch in looper:
        running_loss = 0
        val_loss = 0
        for inputs, targets in dataloaders[0]:

            inputs, targets = process_data(waveform_transforms, inputs, targets)
            optimizer.zero_grad()
            predictions = model(inputs)
            if logger is not None and "log_ios_train" in dir(logger):
                logger.log_ios_train(inputs, targets, predictions, epoch)
            loss = criterion(predictions, targets)
            if "regularizer" in dir(model):
                loss = loss + model.regularizer()

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_losses.append(running_loss / len(dataloaders[0]))
        description = "Training Loss: {:.6f}.. ".format(train_losses[-1])

        if dataloaders[1] is not None and len(dataloaders[1]) > 0:
            with torch.no_grad():
                model.eval()
                for inputs, targets in dataloaders[1]:
                    inputs, targets = process_data(waveform_transforms, inputs, targets)
                    predictions = model(inputs)
                    if logger is not None and "log_ios_val" in dir(logger):
                        logger.log_ios_val(inputs, targets, predictions)
                    val_loss += criterion(predictions, targets).item()

            model.train()

            val_losses.append(val_loss / len(dataloaders[1]))
            description += "Test Loss: {:.6f}.. ".format(val_losses[-1])
            if save_dir is not None and val_losses[-1] < min_val_loss:
                min_val_loss = val_losses[-1]
                description += " Saving model ..."
                torch.save(model, os.path.join(save_dir, "model.pt"))
        looper.set_description(description)
        if logger is not None and "log_val_predictions" in dir(logger):
            logger.log_performance(train_losses, val_losses, epoch)

        # TODO: Add a save instruction and a stopping criteria
        # if stopping_criteria(train_losses, val_losses):
        #     break

    if logger is not None:
        logger.close()
    if (
        save_dir is not None
        and return_best_model
        and dataloaders[1] is not None
        and len(dataloaders[1]) > 0
    ):
        model = torch.load(os.path.join(save_dir, "model.pt"))
    else:
        torch.save(model, os.path.join(save_dir, "model.pt"))
    # print(prof)
    return model, {
        "performance_history": [torch.tensor(train_losses), torch.tensor(val_losses)]
    }
