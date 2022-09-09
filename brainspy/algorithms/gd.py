import os
import torch
import numpy as np
from tqdm import trange
from brainspy.utils.pytorch import TorchUtils
from torch.utils.data import DataLoader


def train(
    model: torch.nn.Module,
    dataloaders: list,
    criterion,
    optimizer: torch.optim.Optimizer,
    configs: dict,
    logger=None,
    save_dir: str = None,
    return_best_model: bool = True,
):
    """
    Main training loop for off-chip gradient descent training  with early stopping using PyTorch.
    It is a default training loop used for simple training tasks, but its code can be taken as a
    reference on how to implement a training loop for more specific or complext tasks.

    Parameters
    ----------
    model : torch.nn.Module
        The model to be trained. It should be an instance of a torch.nn.Module. It can be a
        Processor, representing a hardware DNPU or a DNPU model, but it also can be a model that
        contains different more complex architectures using several processors.

        - The model can have multiple DNPU instances.
        - The model cannot be an instance of SurrogateModel or HardwareProcessor.
        - The model should have the following methods implemented :

        1. format_targets : The hardware processor uses a waveform to represent points
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

        2. set_regul_factor : This method only needs to be implemented if
                              constraint_control_voltages = "regul" in the configs (see description below)

                              Parameters
                              ----------
                              regul_factor : int
                                            See description below

        3. regularizer : This method only needs to be implemented if
                         constraint_control_voltages = "regul" in the configs (see description below)
                         An example can be found at: brainspy.processors.dnpu, inside the class DNPU

                         Parameters : None

        4. constraint_weights : This method only needs to be implemented if
                         constraint_control_voltages = "clip" in the configs (see description below)

                         Parameters : None


    dataloaders : list
        A list containing one or two Pytorch dataloaders. The first dataloader corresponds to the
        training dataset. The second dataloader is optional, and it corresponds to the validation
        dataset. If no validation dataset is given, the training loop will train the model and
        return the trained model only after reaching to the latest epoch. If a second dataloader is
        given, it will be used as a validation dataset. When a validation dataset is present, only
        models with solutions that achieve the lowest validation score will be saved. It is
        recommended to have an additional test dataset on the side, to check the model against,
        after training it with an additional validation datasetz

        More information about dataloaders can be found at:
        https://pytorch.org/tutorials/beginner/basics/data_tutorial.html

    criterion : Object <method>
        Loss function criterion that will be used to optimise the model. More information on
        several loss functions supported can be found at:
        https://pytorch.org/docs/stable/nn.html#loss-functions

    optimizer : torch.optim.Optimizer
        Optimisation algorithm to be used during the training process. More on Pytorch's optimizer
        package can be found at:
        https://pytorch.org/docs/stable/optim.html

    configs : dict
        Dictionary containing the following extra configuration keys:
            epochs : int
                Number of passes through the entire training dataset.
            constraint_control_voltages : str
                When training models, typically it is desired for the control voltages to stay
                within the ranges in which they where trained, in order to avoid extrapolating, or
                reaching the clipping values. This str key can have the following values:
                    'regul' : It applies a penalty to the loss function when control voltages go
                              outside the  ranges in which they were trained. This method allows a
                              bit of flexibility, enabling to find solutions that are, in some
                              cases, slightly outside of the control voltage ranges. In order to be
                              used, it also requires that the model has a method called
                              'regularizer' which controls that penalty. An example can be found at:
                              brainspy.processors.dnpu, inside the class DNPU, method regularizer.
                    'clip' : It applies clipping after the backward pass and optimiser step. It
                             enforces that the control voltage ranges will not be outside the
                             ranges in which the model was trained. In order to use it, the model
                             should have a method called 'constraint_weights'. An example can be
                             found at: brainspy.processors.dnpu, inside the class DNPU, method
                             constraint_weights.

    logger: logging (optional)
        It provides a way for applications to configure different log handlers.
        by default None.
        The logger should be an already initialised class that contains a method called
        'log_output', where the input is a single numpy array variable. It can be any class,
        and the data can be treated in the way the user wants.You can get more information about
        loggers at https://pytorch.org/docs/stable/tensorboard.html

        Logger directory info :
            log_train_step: to log each step in the training process
            log_val_step: to log each step in the validation process

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
        Trained model with best results according to the criterion fitness function.
    training_data: dict
        Dictionary returning relevant data produced while training the model.

    configs['return_best_model']: boolean
        It also adds to the configs dictionary whether the algorithm was returning the best model or not
        at configs['return_best_model'].

    Saved Data
    ----------
    A) After the end of the last epoch, the algorithm saves two main files:
        model_raw.pt: An exact copy of the model after the end of the training process. It can be loaded directly as
        an instance of the model using:
            my_model_instance_at_best_val_results = torch.load('best_model_raw.pt').
        training_data.pickle: A pytorch picle which contains the following keys:
            - epochs: int
                Number of epochs used for training the model
            - algorithm:
                Algorithm type that was being used. Either 'genetic' or 'gradient'.
            - optimizer_state_dict: OrderedDict
                State of the optimizer at the end of last epoch. It can be used to resume model training at that
                exact point.
            - model_state_dict: OrderedDict
                It contains the value of the learnable parameters (weights, or in this case, control voltages) at
                the point where all the training was finised.
            - train_losses: list
                A list of the loss performance over all epochs
            - val_losses: list
                A list of the loss performance over all epochs
    B) If there is a validation dataset present, and return_best_model is set to true. The algorithm will
    save, each time that the validation loss is better than the previous, the following files:
        best_model_raw.pt: An exact copy of the model when it got the best validation results. It can be
        loaded directly as an instance of the model using:
                            my_model_instance_at_best_val_results = torch.load('best_model_raw.pt').
        best_training_data.pickle: A pytorch picle which contains the following keys:
            - epoch: int
                Epoch at which the model with best validation loss was found.
            - algorithm: str
                Algorithm type that was being used. Either 'genetic' or 'gradient'.
            - optimizer_state_dict: OrderedDict
                State of the optimizer at the moment when the best validation loss was achieved. It can be used
                to resume model training at that exact point.
            - model_state_dict: OrderedDict
                It contains the value of the learnable parameters (weights, or in this case, control voltages) at
                the point where the best validation was achieved.
            - train_loss: float
                Training loss at the point where the best validation was achieved.
            - validation_loss: float
                Best validation loss achieved.
    """
    train_checks(model, dataloaders, criterion, optimizer, configs, save_dir)

    start_epoch = 0
    train_losses, val_losses = [], []
    min_val_loss = np.inf

    looper = trange(configs["epochs"], desc=" Initialising")
    looper.update(start_epoch)
    configs['return_best_model'] = return_best_model
    model.to(device=TorchUtils.get_device())

    for epoch in looper:

        model, running_loss = default_train_step(
            model,
            epoch,
            dataloaders[0],
            criterion,
            optimizer,
            logger=logger,
            constraint_control_voltages=configs['constraint_control_voltages'])
        train_losses.append(running_loss)
        description = "Training Loss: {:.6f}.. ".format(train_losses[-1])

        if len(dataloaders) > 1 and dataloaders[1] is not None and len(
                dataloaders[1]) > 0:
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
            if save_dir is not None and (val_losses[-1] < min_val_loss
                                         or epoch == 0):
                min_val_loss = val_losses[-1]
                description += " Saving model ..."
                torch.save(model, os.path.join(save_dir, "best_model_raw.pt"))
                torch.save(
                    {
                        "epochs": epoch,
                        "algorithm": 'gradient',
                        "optimizer_state_dict": optimizer.state_dict(),
                        "model_state_dict": model.state_dict(),
                        "train_loss": train_losses[-1],
                        "val_loss": val_losses[-1],
                    },
                    os.path.join(save_dir, "best_training_data.pickle"),
                )

        looper.set_description(description)
        if logger is not None and "log_performance" in dir(logger):
            logger.log_performance(train_losses, val_losses, epoch)

        # TODO: Add a save instruction and a stopping criteria
        # if stopping_criteria(train_losses, val_losses):
        #     break
    if save_dir is not None:
        torch.save(
            model,
            os.path.join(
                save_dir,  # type: ignore[arg-type]
                "model_raw.pt"))
        torch.save(
            {
                "epoch": epoch + 1,
                "algorithm": 'gradient',
                "optimizer_state_dict": optimizer.state_dict(),
                "model_state_dict": model.state_dict(),
                "train_losses": train_losses,
                "val_losses": val_losses,
                "min_val_loss": min_val_loss,
            },
            os.path.join(
                save_dir,  # type: ignore[arg-type]
                "training_data.pickle"),
        )
    if logger is not None:
        logger.close()
    if (save_dir is not None and return_best_model
            and (len(dataloaders) == 1 or
                 (dataloaders[1] is not None and len(dataloaders[1]))) > 0):
        if os.path.exists(os.path.join(save_dir, "best_model_raw.pt")):
            model = torch.load(os.path.join(save_dir, "best_model_raw.pt"))

    return model, {
        "performance_history":
        [torch.tensor(train_losses),
         torch.tensor(val_losses)]
    }


def train_checks(model, dataloaders, criterion, optimizer, configs, save_dir):
    """ Performs several assertions over the parameters that enter the train function.

    Parameters
    ----------
    model : torch.nn.Module
        The model to be trained. It should be an instance of a torch.nn.Module. It can be a
        Processor, representing a hardware DNPU or a DNPU model, but it also can be a model that
        contains different more complex architectures using several processors.

        - The model can have multiple DNPU instances.
        - The model cannot be an instance of SurrogateModel or HardwareProcessor.
        - The model should have the following methods implemented :

        1. format_targets : The hardware processor uses a waveform to represent points
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

        2. set_regul_factor : This method only needs to be implemented if
                              constraint_control_voltages = "regul" in the configs (see description below)

                              Parameters
                              ----------
                              regul_factor : int
                                            See description below

        3. regularizer : This method only needs to be implemented if
                         constraint_control_voltages = "regul" in the configs (see description below)
                         An example can be found at: brainspy.processors.dnpu, inside the class DNPU

                         Parameters : None

        4. constraint_weights : This method only needs to be implemented if
                         constraint_control_voltages = "clip" in the configs (see description below)

                         Parameters : None


    dataloaders : list
        A list containing one or two Pytorch dataloaders. The first dataloader corresponds to the
        training dataset. The second dataloader is optional, and it corresponds to the validation
        dataset. If no validation dataset is given, the training loop will train the model and
        return the trained model only after reaching to the latest epoch. If a second dataloader is
        given, it will be used as a validation dataset. When a validation dataset is present, only
        models with solutions that achieve the lowest validation score will be saved. It is
        recommended to have an additional test dataset on the side, to check the model against,
        after training it with an additional validation datasetz

        More information about dataloaders can be found at:
        https://pytorch.org/tutorials/beginner/basics/data_tutorial.html

    criterion : Object <method>
        Loss function criterion that will be used to optimise the model. More information on
        several loss functions supported can be found at:
        https://pytorch.org/docs/stable/nn.html#loss-functions

    optimizer : torch.optim.Optimizer
        Optimisation algorithm to be used during the training process. More on Pytorch's optimizer
        package can be found at:
        https://pytorch.org/docs/stable/optim.html

    configs : dict
        Dictionary containing the following extra configuration keys:
            epochs : int
                Number of passes through the entire training dataset.
            constraint_control_voltages : str
                When training models, typically it is desired for the control voltages to stay
                within the ranges in which they where trained, in order to avoid extrapolating, or
                reaching the clipping values. This str key can have the following values:
                    'regul' : It applies a penalty to the loss function when control voltages go
                              outside the  ranges in which they were trained. This method allows a
                              bit of flexibility, enabling to find solutions that are, in some
                              cases, slightly outside of the control voltage ranges. In order to be
                              used, it also requires that the model has a method called
                              'regularizer' which controls that penalty. An example can be found at:
                              brainspy.processors.dnpu, inside the class DNPU, method regularizer.
                    'clip' : It applies clipping after the backward pass and optimiser step. It
                             enforces that the control voltage ranges will not be outside the
                             ranges in which the model was trained. In order to use it, the model
                             should have a method called 'constraint_weights'. An example can be
                             found at: brainspy.processors.dnpu, inside the class DNPU, method
                             constraint_weights.

    logger: logging (optional)
        It provides a way for applications to configure different log handlers.
        by default None.
        The logger should be an already initialised class that contains a method called
        'log_output', where the input is a single numpy array variable. It can be any class,
        and the data can be treated in the way the user wants.You can get more information about
        loggers at https://pytorch.org/docs/stable/tensorboard.html

        Logger directory info :
            log_train_step: to log each step in the training process
            log_val_step: to log each step in the validation process

    save_dir : Optional[str]
        Folder where the trained model is going to be saved.
        When None, the model will not be saved.
        By default None.

    return_best_model : bool, optional
        to return the trained model instead of saving
        it to a directory, by default True
    """
    assert isinstance(
        model,
        torch.nn.Module), "The model should be an instance of torch.nn.Module"
    assert "format_targets" in dir(
        model), "The format_targets function should be implemeted in the model"
    assert type(
        dataloaders) == list, "The dataloaders should be of type - list"
    for dataloader in dataloaders:
        assert isinstance(
            dataloader, DataLoader
        ), "The dataloader should be an instance of torch.utils.data.DataLoader"
    assert callable(criterion), "The criterion should be a callable method"
    assert isinstance(
        optimizer, torch.optim.Optimizer
    ), "The optimizer object should be an instance of torch.optim.Optimizer"
    assert type(configs) == dict, "The extra configs should be of type - dict"
    if configs["epochs"]:
        assert type(
            configs["epochs"]) == int, "The epochs key should be of type - int"
    assert type(
        configs["constraint_control_voltages"]
    ) == str, "The constraint_control_voltages key should be of type str"
    assert configs["constraint_control_voltages"] == "clip" or configs[
        "constraint_control_voltages"] == "regul", "The constraint_control_voltages should be either clip or regul"
    if configs["constraint_control_voltages"] == "regul":
        assert "regularizer" in dir(
            model
        ), "The model should implement the regularizer function for this option"
    else:
        assert "constraint_weights" in dir(
            model
        ), "The model should implement the constraint_weights function for this option"
    assert save_dir is None or type(
        save_dir
    ) == str, "The name/path of the save_dir should be of type - str"


def default_train_step(model,
                       epoch,
                       dataloader,
                       criterion,
                       optimizer,
                       logger=None,
                       constraint_control_voltages=None):
    """
    Deafult training step for training a torch model in Gradiet descent. The method calulates the
    training loss in each training step. The training loss indicates how well the model is fitting
    the training data.

    More information about training loss can be found at
    https://www.baeldung.com/cs/learning-curve-ml

    The method returns the trained model and the running loss, which is used to calculate the
    training loss, in that step.

    Parameters
    ----------
    model : torch.nn.Module
        The model to be trained. It should be an instance of a torch.nn.Module. It can be a
        Processor, representing a hardware DNPU or a DNPU model, but it also can be a model that
        contains different more complex architectures using several processors. Refer to the
        documentation of the train function above for more inforamtion about defining a model.
    epoch : int
        Number of passes through the entire training dataset.
    dataloader : torch.utils.data.Dataloader
        A Pytorch dataloaders that corresponds to the training dataset.
        More information about dataloaders can be found at:
        https://pytorch.org/tutorials/beginner/basics/data_tutorial.html

    criterion : Object <method>
        Loss function criterion that will be used to optimise the model. More information on
        several loss functions supported can be found at:
        https://pytorch.org/docs/stable/nn.html#loss-functions

    optimizer : torch.optim.Optimizer
        Optimisation algorithm to be used during the training process. More on Pytorch's optimizer
        package can be found at:
        https://pytorch.org/docs/stable/optim.html

    logger: logging (optional)
        It provides a way for applications to configure different log handlers.
        by default None.
        The logger should be an already initialised class that contains a method called
        'log_output', where the input is a single numpy array variable. It can be any class,
        and the data can be treated in the way the user wants.You can get more information about
        loggers at https://pytorch.org/docs/stable/tensorboard.html

        Logger directory info :
            log_train_step: to log each step in the training process

    constraint_control_voltages : str
        When training models, typically it is desired for the control voltages to stay
        within the ranges in which they where trained, in order to avoid extrapolating, or
        reaching the clipping values. This str key can have the following values:
            'regul' : It applies a penalty to the loss function when control voltages go
                        outside the  ranges in which they were trained. This method allows a
                        bit of flexibility, enabling to find solutions that are, in some
                        cases, slightly outside of the control voltage ranges. In order to be
                        used, it also requires that the model has a method called
                        'regularizer' which controls that penalty. An example can be found at:
                        brainspy.processors.dnpu, inside the class DNPU, method regularizer.
            'clip' : It applies clipping after the backward pass and optimiser step. It
                        enforces that the control voltage ranges will not be outside the
                        ranges in which the model was trained. In order to use it, the model
                        should have a method called 'constraint_weights'. An example can be
                        found at: brainspy.processors.dnpu, inside the class DNPU, method
                        constraint_weights.

    Returns
    -------
    model : torch.nn.Module
        Trained model with best results according to the criterion fitness function.
    running loss : int
        To assess the training loss: how far the predictions of the model are from the actual
        targets.
    """
    assert isinstance(
        model,
        torch.nn.Module), "The model should be an instance of torch.nn.Module"
    assert "format_targets" in dir(
        model), "The format_targets function should be implemeted in the model"
    assert type(epoch) == int, "The epoch param should be of type - int"
    assert isinstance(
        dataloader, DataLoader
    ), "The dataloader should be an instance of torch.utils.data.DataLoader"
    assert callable(criterion), "The criterion should be a callable method"
    assert isinstance(
        optimizer, torch.optim.Optimizer
    ), "The optimizer should be an instance of torch.optim.Optimizer"
    assert (
        constraint_control_voltages is None
        or constraint_control_voltages == "clip"
        or constraint_control_voltages == "regul"
    ), "The constraint_control_voltages should be None or 'clip' or 'regul'"
    if constraint_control_voltages == "regul":
        assert "regularizer" in dir(
            model
        ), "The model should implement the regularizer function for this option"
    else:
        assert "constraint_weights" in dir(
            model
        ), "The model should implement the constraint_weights function for this option"
    assert (
        constraint_control_voltages is None
        or constraint_control_voltages == 'clip'
        or constraint_control_voltages == 'regul'
    ), "Variable constraint_control_voltages should be 'regul', 'clip' or None."
    running_loss = 0
    model.train()
    for inputs, targets in dataloader:
        inputs, targets = TorchUtils.format(inputs), model.format_targets(
            TorchUtils.format(targets))

        optimizer.zero_grad()
        predictions = model(inputs)

        if constraint_control_voltages is None or constraint_control_voltages == 'clip':
            loss = criterion(predictions, targets)
        elif constraint_control_voltages == 'regul':
            loss = criterion(predictions, targets) + model.regularizer()

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
    """
    To calulate the validation loss in each training step of the Gradient descent.
    Validation loss indicates how well the model fits unseen data.
    More information about validation loss and training loss can be
    found at https://www.baeldung.com/cs/learning-curve-ml

    Parameters
    ----------
    epoch : int
        Number of passes through the entire training dataset.
    model : torch.nn.Module
        The model to be trained. It should be an instance of a torch.nn.Module. It can be a
        Processor, representing a hardware DNPU or a DNPU model, but it also can be a model that
        contains different more complex architectures using several processors.Refer to the
        documentation of the train function above for more inforamtion about defining a model.
    dataloader : torch.utils.data.Dataloader
        A Pytorch dataloaders that corresponds to the validation dataset.
        More information about dataloaders can be found at:
        https://pytorch.org/tutorials/beginner/basics/data_tutorial.html

    criterion : Object <method>
        Loss function criterion that will be used to optimise the model. More information on
        several loss functions supported can be found at:
        https://pytorch.org/docs/stable/nn.html#loss-functions

    logger: logging (optional)
        It provides a way for applications to configure different log handlers.
        by default None.
        The logger should be an already initialised class that contains a method called
        'log_output', where the input is a single numpy array variable. It can be any class,
        and the data can be treated in the way the user wants.You can get more information about
        loggers at https://pytorch.org/docs/stable/tensorboard.html

        Logger directory info :
            log_val_step: to log each step in the validation process

    Returns
    -------
    val_loss : int
        To assess how well the model fits new data.
        It is the sum of errors made for each example in training or validation sets.
    """
    assert isinstance(
        model,
        torch.nn.Module), "The model should be an instance of torch.nn.Module"
    assert "format_targets" in dir(
        model), "The format_targets function should be implemeted in the model"
    assert type(epoch) == int, "The epoch param should be of type - int"
    assert isinstance(
        dataloader, DataLoader
    ), "The dataloader should be an instance of torch.utils.data.DataLoader"
    assert callable(criterion), "The criterion should be a callable method"
    with torch.no_grad():
        val_loss = 0
        model.eval()
        for inputs, targets in dataloader:
            inputs, targets = TorchUtils.format(inputs), model.format_targets(
                TorchUtils.format(targets))
            predictions = model(inputs)
            loss = criterion(predictions, targets).item()
            val_loss += loss * inputs.shape[0]
            if logger is not None and "log_val_step" in dir(logger):
                logger.log_val_step(epoch, inputs, targets, predictions, model,
                                    loss, val_loss)
        val_loss /= len(dataloader.dataset)
    return val_loss
