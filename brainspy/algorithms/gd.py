import os
import torch
import numpy as np
from tqdm import trange
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
    """
    Main training loop for the Gradient descent. It supports training
    a single DNPU hardware device on both on and off chip flavours.
    It supports using both a training dataset and validation dataset.

    More information about Gradient descent can be found at
    https://towardsdatascience.com/gradient-descent-explained-9b953fc0d2c

    Parameters
    ----------
    model : torch.nn.Module
        Model to be trained.
    dataloaders : list
                  A list containing a single PyTorch Dataloader containing the
                  training dataset.
                  More information about dataloaders can be found at:
                  https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
    criterion : <method>
                Fitness function that will be used to train the model.
    optimizer : torch.optim
                Optimization method for sorting the genome pool by fitness and
                creating new offspring based on the best resulting genomes.
    configs : dict
        A dictionary containing extra configurations for the algorithm.
            * epochs : int
                Number of steps (generations) that the algorithm will take in
                order to train the model.
            * constraint_control_voltages : str
                applies the models regulaizer or clip.
                Options :
                    'regul' - This option allows a bit of freedom to go
                              outside the voltage ranges, where the NN
                              model would be extrapolating.
                    'clip' -  forces to remain within the
                              control_voltage_ranges
            * regul_factor: int, Optional
                If the 'regul' option is chosen for the constraint
                control_voltages,the regul_factor is can be set with a
                bit of freedom to go outside the voltage ranges, where
                the NN model would be extrapolating.

    logger: logging (optional) - It provides a way for applications to
                                 configure different log handlers ,
                                 by default None.

            The logger should be an already initialised class that contains
            a method called 'log_output', where the input is a single numpy
            array variable. It can be any class, and the data can be treated
            in the way the user wants.You can get more information about
            loggers at https://pytorch.org/docs/stable/tensorboard.html
            Logger directory info :
                log_train_step: to log each step in the training process

    save_dir : Optional[str]
        Folder where the trained model is going to be saved.
        When None, the model will not be saved.
        By default None.

    return_best_model : bool, optional
        to return the trained model instead of saving
        it to a directory, by default True

    Returns
    -------
    model : torch.nn.Module
        Trained model with best results according to the criterion 
        fitness function.

    training_data: dict
        Dictionary returning relevant data produced while training
        the model.
    """

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
            constraint_control_voltages=configs["constraint_control_voltages"],
        )
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
    return model, {
        "performance_history":
        [torch.tensor(train_losses),
         torch.tensor(val_losses)]
    }


def default_train_step(
    model,
    dataloader,
    criterion,
    optimizer,
    logger=None,
    constraint_control_voltages=None,
):
    """
    Deafult training step for training the model in Gradiet descent.

    The method calulates the training loss in each training step.
    The training loss indicates how well the model is fitting the
    training data.
    More information about training loss can be found at
    https://www.baeldung.com/cs/learning-curve-ml

    The method returns the trained model and the running loss
    , which is used to calculate the training loss, in that step.

    Parameters
    ----------
     model : torch.nn.Module
        Model to be trained.
    dataloaders : list
                  A list containing a single PyTorch Dataloader
                  containing the training dataset.
                  More information about dataloaders can be found at:
                  https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
    criterion : <method>
                Fitness function that will be used to train the model.
    optimizer : torch.optim
                Optimization method for sorting the genome pool by fitness and
                creating new offspring based on the best resulting genomes.
    logger: logging (optional) - It provides a way for applications to
                                 configure different log handlers ,
                                 by default None.

            The logger should be an already initialised class that contains
            a method called 'log_output', where the input is a single numpy
            array variable. It can be any class, and the data can be treated
            in the way the user wants.You can get more information about
            loggers at https://pytorch.org/docs/stable/tensorboard.html
            Logger directory info :
                log_train_step: to log each step in the training process

    constraint_control_voltages : str, optional
        applies the models regulaizer or clip.
                Options :
                    'regul' - This option allows a bit of freedom to go
                              outside the voltage ranges, where the
                              NN model would be extrapolating.
                    'clip' -  forces to remain within the control_voltage
                             _ranges,
                    by default None

    Returns
    -------
    model : torch.nn.Module
        Trained model with best results according to the criterion
        fitness function.

    running loss : int
        To assess the training loss: how far the predictions of
        the model are from the actual targets.
    """
    running_loss = 0
    model.train()
    for inputs, targets in dataloader:
        inputs, targets = TorchUtils.format(inputs), model.format_targets(
            TorchUtils.format(targets))

        optimizer.zero_grad()
        predictions = model(inputs)

        if constraint_control_voltages is None or constraint_control_voltages == "clip":
            loss = criterion(predictions, targets)
        elif constraint_control_voltages == "regul":
            loss = criterion(predictions, targets) + model.regularizer()
        else:
            # TODO Throw an error adequately
            assert (
                False
            ), "Constraint_control_voltages variable should be either 'regul', 'clip' or None. "

        loss.backward()
        optimizer.step()

        if (constraint_control_voltages is not None
                and constraint_control_voltages == "clip"):
            #    with torch.no_grad():
            model.constraint_weights()

        running_loss += loss.item() * inputs.shape[0]
        if logger is not None and "log_train_step" in dir(logger):
            logger.log_train_step(epoch, inputs, targets, predictions, model,
                                  loss, running_loss)

    running_loss /= len(dataloader.dataset)
    return model, running_loss


def default_val_step(epoch, model, dataloader, criterion, logger=None):
    """
    To calulate the validation loss in each training step of the
    Gradient descent.

    Validation loss indicates how well the model fits new data.
    More information about validation loss and training loss can be
    found at https://www.baeldung.com/cs/learning-curve-ml

    Parameters
    ----------
    epochs : int
            Number of steps (generations) that the algorithm will take
            in order to train the model.
    model : torch.nn.Module
        Model to be trained.
    dataloader : list
              A list containing a single PyTorch Dataloader containing
              the training dataset.
              More information about dataloaders can be found at:
              https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
    criterion : <method>
                Fitness function that will be used to train the model.
    logger: logging (optional) - It provides a way for applications to
                                 configure different log handlers ,
                                 by default None.

            The logger should be an already initialised class that contains
            a method called 'log_output', where the input is a single numpy
            array variable. It can be any class, and the data can be treated
            in the way the user wants.You can get more information about
            loggers at https://pytorch.org/docs/stable/tensorboard.html
            Logger directory info :
                log_train_step: to log each step in the training process


    Returns
    -------
    val_loss: int
        value indicating how well the model fits new data
    """
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
