from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

def get_scheduler(
    optimizer,
    lr_scheduler : str,
    lr_scheduler_decay_steps : int,
    lr_scheduler_decay_rate : float,
    lr_scheduler_patience : int
    ):
    if lr_scheduler:
        if lr_scheduler == 'StepLR':
            scheduler = StepLR(
                optimizer, 
                step_size=lr_scheduler_decay_steps,
                gamma=lr_scheduler_decay_rate
            )
        elif lr_scheduler == "ReduceLROnPlateau":
            scheduler = ReduceLROnPlateau(
                optimizer, 
                mode='min',
                factor=lr_scheduler_decay_rate,
                patience=lr_scheduler_patience,
                verbose=False
            )
        else:
            raise NotImplementedError(
                f'Scheduler {lr_scheduler} is not currently supported.'
            )
    else:
        scheduler = None
    return scheduler

def step_scheduler(
    scheduler, 
    lr_scheduler, 
    val_loss
):
    """
    Steps the learning rate scheduler forward by one
    """
    if lr_scheduler:
        if lr_scheduler == 'StepLR':
            scheduler.step()
        elif lr_scheduler == "ReduceLROnPlateau":
            scheduler.step(val_loss)