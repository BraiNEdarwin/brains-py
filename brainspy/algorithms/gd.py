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
            logger=logger,
            constraint_control_voltages=configs['constraint_control_voltages'])
        train_losses.append(running_loss)
        description = "Training Loss: {:.6f}.. ".format(train_losses[-1])

        if dataloaders[1] is not None and len(dataloaders[1]) > 0:
            val_loss = default_val_step(
                epoch,
                model,
                dataloaders[1],
                criterion,
                logger=logger,
            )
            val_losses.append(val_loss)
            description += "Validation Loss: {:.6f}.. ".format(val_losses[-1])
            # Save only when peak val performance is reached
            if save_dir is not None and val_losses[-1] < min_val_loss:
                min_val_loss = val_losses[-1]
                description += " Saving model ..."
                torch.save(model, os.path.join(save_dir, "model.pt"))
                torch.save(
                    {
                        "epoch": epoch,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "model_state_dict": model.state_dict(),
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
    if (save_dir is not None and return_best_model
            and dataloaders[1] is not None and len(dataloaders[1]) > 0):
        model = torch.load(os.path.join(save_dir, "model.pt"))
    else:
        torch.save(model, os.path.join(save_dir, "model.pt"))
        torch.save(
            {
                "epoch": epoch,
                "optimizer_state_dict": optimizer.state_dict(),
                "model_state_dict": model.state_dict(),
                "train_losses": train_losses,
                "val_losses": val_losses,
                "min_val_loss": min_val_loss,
            },
            os.path.join(save_dir, "training_data.pickle"),
        )
    # print(prof)
    return model, {
        "performance_history":
        [torch.tensor(train_losses),
         torch.tensor(val_losses)]
    }


# Constraint_control_voltages can either be 'regul' to apply the models regularizer, or 'clip'. The first option allows a bit of freedom to go outside the voltage ranges, where the NN model would be extrapolating.
# The second option forces to remain within the control_voltage_ranges.
def default_train_step(model,
                       dataloader,
                       criterion,
                       optimizer,
                       logger=None,
                       constraint_control_voltages=None):
    running_loss = 0
    model.train()
    for inputs, targets in dataloader:
        inputs, targets = TorchUtils.format(inputs), model.format_targets(
            TorchUtils.format(targets))

        optimizer.zero_grad()
        #
        predictions = model(inputs)

        if constraint_control_voltages is None or constraint_control_voltages == 'clip':
            loss = criterion(predictions, targets)
        elif constraint_control_voltages == 'regul':
            loss = criterion(predictions, targets) + model.regularizer()
        else:
            #TODO Throw an error adequately
            assert False, "Constraint_control_voltages variable should be either 'regul',  'clip' or None. "

        loss.backward()
        optimizer.step()

        if constraint_control_voltages is not None and constraint_control_voltages == 'clip':
            #    with torch.no_grad():
            model.constraint_weights()

        running_loss += loss.item() * inputs.shape[0]
        if logger is not None and "log_train_step" in dir(logger):
            logger.log_train_step(epoch, inputs, targets, predictions, model,
                                  loss, running_loss)

    running_loss /= len(dataloader.dataset)
    return model, running_loss


def default_val_step(epoch, model, dataloader, criterion, logger=None):
    with torch.no_grad():
        val_loss = 0
        model.eval()
        for inputs, targets in dataloader:
            inputs, targets = process_data(inputs, targets)
            targets = model.format_targets(targets)
            predictions = model(inputs)
            loss = criterion(predictions, targets).item()
            val_loss += loss * inputs.shape[0]
            if logger is not None and "log_val_step" in dir(logger):
                logger.log_val_step(epoch, inputs, targets, predictions, model,
                                    loss, val_loss)
        val_loss /= len(dataloader.dataset)
    return val_loss


# def format_data(inputs, targets):
#     # Data processing required to apply waveforms to the inputs and pass them
#     # onto the GPU if necessary.
#     # if waveform_transforms is not None:
#     #     inputs, targets = waveform_transforms((inputs, targets))
#     if inputs is not None and inputs.device != TorchUtils.get_device():
#         inputs = inputs.to(device=TorchUtils.get_device())
#     if targets is not None and targets.device != TorchUtils.get_device():
#         targets = targets.to(device=TorchUtils.get_device())

#     return inputs, targets