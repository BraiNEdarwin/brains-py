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

    start_epoch = 0
    train_losses, val_losses = [], []
    min_val_loss = np.inf

    looper = trange(configs["epochs"], desc=" Initialising")
    looper.update(start_epoch)
    model.to(device=TorchUtils.get_device())

    if "set_regul_factor" in dir(model) and "regul_factor" in configs:
        model.set_regul_factor(configs["regul_factor"])

    for epoch in looper:

        model, running_loss = default_train_step(
            model,
            dataloaders[0],
            criterion,
            optimizer,
            waveform_transforms=waveform_transforms,
            logger=logger,
        )
        train_losses.append(running_loss)
        description = "Training Loss: {:.6f}.. ".format(train_losses[-1])

        if dataloaders[1] is not None and len(dataloaders[1]) > 0:
            val_loss = default_val_step(
                model,
                dataloaders[1],
                criterion,
                waveform_transforms=waveform_transforms,
                logger=logger,
            )
            val_losses.append(val_loss)
            description += "Test Loss: {:.6f}.. ".format(val_losses[-1])
            # Save only when peak val performance is reached
            if save_dir is not None and val_losses[-1] < min_val_loss:
                min_val_loss = val_losses[-1]
                description += " Saving model ..."
                torch.save(model, os.path.join(save_dir, "model.pt"))
                torch.save(
                    {
                        "epoch": epoch,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "train_losses": train_losses,
                        "val_losses": val_losses,
                        "min_val_loss": min_val_loss,
                    },
                    os.path.join(save_dir, "training_data.pickle"),
                )

        looper.set_description(description)
        if logger is not None and "log_performance" in dir(logger):
            logger.log_performance(train_losses, val_losses, epoch)

        # TODO: Add a save instruction and a stopping criteria
        # if stopping_criteria(train_losses, val_losses):
        #     break

    if logger is not None:
        logger.close()
    if (
        save_dir is not None and return_best_model and dataloaders[1] is not None and len(dataloaders[1]) > 0
    ):
        model = torch.load(os.path.join(save_dir, "model.pt"))
    else:
        torch.save(model, os.path.join(save_dir, "model.pt"))
    # print(prof)
    return model, {
        "performance_history": [torch.tensor(train_losses), torch.tensor(val_losses)]
    }


def default_train_step(
    model, dataloader, criterion, optimizer, waveform_transforms=None, logger=None
):
    train_loss = 0
    model.train()
    for inputs, targets in dataloader:
        inputs, targets = process_data(waveform_transforms, inputs, targets)
        optimizer.zero_grad()
        predictions = model(inputs)

        if "regularizer" in dir(model):
            loss = criterion(predictions, targets) + model.regularizer()
        else:
            loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()
        if logger is not None and "log_train_step" in dir(logger):
            logger.log_train_step(
                epoch, inputs, targets, predictions, model, loss, running_loss
            )
        train_loss += loss.item()
    train_loss /= len(dataloader)
    return model, train_loss


def default_val_step(
    model, dataloader, criterion, waveform_transforms=None, logger=None
):
    with torch.no_grad():
        val_loss = 0
        model.eval()
        for inputs, targets in dataloader:
            inputs, targets = process_data(waveform_transforms, inputs, targets)
            predictions = model(inputs)
            loss = criterion(predictions, targets).item()
            val_loss += loss
            if logger is not None and "log_val_step" in dir(logger):
                logger.log_val_step(
                    epoch, inputs, targets, predictions, model, loss, val_loss
                )
        val_loss /= len(dataloader)
    return val_loss
