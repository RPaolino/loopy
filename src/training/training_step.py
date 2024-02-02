import torch
import torch.optim

def initialize_tracking_dict(
    metric_name
):
    tracking_dict = {
        "train": {
            "loss": float("nan"),
            metric_name: float("nan")
        },
        "val": {
            "loss": float("nan"),
            metric_name: float("nan")
        },
        "test": {
            "loss": float("nan"),
            metric_name: float("nan")
        },
    }
    return tracking_dict

def training_step(
    dataset_name,
    mean,
    std,
    model, 
    optimizer, 
    loss_fn,
    train_loader
):
    """
        Evaluates a model over all the batches of a data loader.
    """
    # Training
    if train_loader is not None:
        model.train()
        for step, batch in enumerate(train_loader):
            #print("STEP: " , step, ": ", batch)
            optimizer.zero_grad()
            y_pred = model(batch)
            y_true = batch.y
            if (dataset_name.startswith("subgraphcount")
                  or dataset_name.startswith("qm9")):
                y_true = (y_true-mean)/std
            loss = loss_fn(y_pred, y_true)
            loss.backward()
            optimizer.step()

def eval_step(
    dataset_name,
    mean,
    std,
    model, 
    loss_fn,
    loaders,
    metric_name, 
    metric_fn,
):
    """
        Evaluates a model over all the batches of a data loader.
    """
    tracking_dict = initialize_tracking_dict(metric_name)
    # Evaluation
    model.eval()
    split = ["train", "val", "test"]
    for n, loader in enumerate(loaders):
        if loader is not None:
            y_pred, y_true = [], []
            for step, batch in enumerate(loader):
                with torch.no_grad():
                    y_pred.append(
                        model(batch)
                    )
                    if (dataset_name.startswith("subgraphcount") 
                          or dataset_name.startswith("qm9")):
                        y_true.append(
                            (batch.y-mean)/std
                        )
                    else:
                        y_true.append(batch.y)
            y_pred = torch.cat(y_pred, dim=0)
            y_true = torch.cat(y_true, dim=0)
            if loss_fn is not None:
                loss = loss_fn(y_pred, y_true)
                tracking_dict[split[n]]["loss"] = loss.item()
            tracking_dict[split[n]][metric_name] = metric_fn(
                y_pred=y_pred, 
                y_true=y_true
            ).item()
    return tracking_dict