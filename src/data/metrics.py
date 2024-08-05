from functools import partial
import numpy as np
from scipy.special import binom
from sklearn.metrics import average_precision_score
import torch
import torch.optim
from torch.nn.functional import l1_loss, cosine_embedding_loss
from typing import Callable

from .dataset_synthetic import Synthetic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
def get_loss(
    dataset_name: str
) -> torch.nn.Module:
    r"""It returns the loss to be used during training on the ``dataset_name`` 
    dataset. 
    """
    dataset_name = dataset_name.lower()
    loss = None
    if (dataset_name in ["zinc",
                        "zinc_subset"]
        or dataset_name.startswith("subgraphcount")
        or dataset_name.startswith("qm9")):
        loss = torch.nn.L1Loss()
    elif dataset_name in ["csl", "exp", "cexp"]:
        loss = torch.nn.CrossEntropyLoss()
    elif dataset_name in ["exp_iso", "cospectral10", "graph8c", "sr16622"]:
        loss = None
    elif dataset_name.startswith("brec"):
        loss = custom_cosine_embedding_loss
    elif dataset_name == "peptides_struct":
        loss = torch.nn.L1Loss()
    elif dataset_name == "peptides_func":
        loss = torch.nn.CrossEntropyLoss()
    else:
        raise NotImplementedError(f'No loss for {dataset_name}.')
    return loss

def get_evaluation_metric(
    dataset_name: str
) -> (str, Callable):
    r"""It returns the function to be used during evaluation on the 
    ``dataset_name`` dataset.
    
    Returns:
        metric_name (str): name of the evaluation metric.
        metric_fn (function): the evaluation metric, with args y_pred, y_true.
    """
    dataset_name = dataset_name.lower()
    metric_fn = None
    if (dataset_name in ["zinc", "zinc_subset"]
        or dataset_name.startswith("subgraphcount")
        or dataset_name.startswith("qm9")):
        metric_name = "mae"
        metric_fn = lambda y_pred, y_true: l1_loss(y_pred, y_true)
    elif dataset_name in ["csl", "exp", "cexp"]:
        metric_name = "acc"
        metric_fn = accuracy_score
    elif dataset_name in ["exp_iso","cospectral10", "graph8c", "sr16622"]:
        metric_name = "num_identical_pairs"
        metric_fn = partial(
            num_identical_pairs,
            dataset_name=dataset_name,
            return_min=(dataset_name in ["sr16622", "cospectral10"])
        )
    elif dataset_name.startswith("brec"):
        metric_name = "cosine_embedding_loss" 
        metric_fn = custom_cosine_embedding_loss
    elif dataset_name == "peptides_struct":
        metric_name = "mae"
        metric_fn = lambda y_pred, y_true: l1_loss(y_pred, y_true)
    elif dataset_name == "peptides_func":
        metric_name = "ap"
        metric_fn = lambda y_pred, y_true: average_precision_score(y_true, y_pred)
    else:
        raise NotImplementedError(f"No evaluation metric for {dataset_name}.")
    return metric_name, metric_fn

def get_task(dataset_name: str) -> str:
    r"""It returns the task to solve on the ``dataset_name`` dataset."""
    if (dataset_name in ["zinc_subset", 
                        "zinc"]
        or dataset_name.startswith("subgraphcount")
        or dataset_name.startswith("qm9")):
        task = "graph_regression"
    elif dataset_name in ["exp","cexp","csl"]:
        task = "graph_classification"   
    elif dataset_name in ["exp_iso", "cospectral10", "graph8c", "sr16622"]:
        task = "num_identical_pairs"
    elif dataset_name.startswith("brec"):
        task = "T^2"
    elif dataset_name == "peptides_struct":
        task = "graph_regression"    
    elif dataset_name == "peptides_func":
        task = "graph_classification"    
    else:
        raise NotImplementedError(f"No task for {dataset_name}") 
    return task

def num_identical_pairs(
    y_pred: torch.Tensor, 
    y_true: torch.Tensor, 
    dataset_name: str, 
    threshold: float = 1e-3,
    normalize: bool = True,
    return_min: bool = False
) -> torch.Tensor:
    r"""
    This function consider two graphs identical if the l1-distance of their embeddings
    is less than ``threshold``. Note that y_true is just a placeholder.
    If ``return_min``, it returns the minimum of pairwise distances.
    
    Args:
        y_pred (Tensor):
            the predicted embedding.
        y_true (Tensor):
            a placeholder to consider this function as an evaluation_metric.
        dataset_name (str):
            name of the dataset.
        threshold (float, optional):
            two graphs for which the L1 distance of their embedding is less than threshold
            are considered identical. Defaults to 1e-3.
        normalize (bool, optional):
            if set to ``True``, normalize the output w.r.t. the number of 
            non-isomorphic pairs. Defaults to ``False``
        return_min (bool, optional):
            if set to ``True``, returns the minimal pairwise L1 distance. 
            Defaults to ``False``
    """
    if dataset_name=="exp_iso":
        pairwise_distances = (y_pred[0::2]-y_pred[1::2]).abs().sum(1)
        out = (pairwise_distances<threshold).sum()
        if normalize:
            out = out/(y_pred.shape[0]/2)
    elif dataset_name=="graph8c":
        idx_1WL_equivalent = np.genfromtxt(
            './equivalent/graph8c_1WL.csv',
            delimiter=',',
            dtype=int
        )
        idx = np.unique(idx_1WL_equivalent).flatten()
        idx = {
            k: r for r, k in enumerate(idx)
        }
        pairwise_distances = torch.tensor([
            (y_pred[idx[i]]-y_pred[idx[j]]).abs().sum().item() for i, j in idx_1WL_equivalent
        ])
        out = (pairwise_distances<threshold).sum()
        if normalize:
            out = out/312
    else:
        num_graphs = y_pred.shape[0]
        pairwise_distances = torch.cat([
            (y_pred[i]-y_pred[i+1:]).abs().sum(1) for i in range(num_graphs-1)
        ])
        assert pairwise_distances.numel()==binom(num_graphs, 2)
        out = (pairwise_distances<threshold).sum()
        if normalize:
            out = out/binom(num_graphs, 2)
    print("Pairwise distances", "(min, mean, max)")
    print("\t min ".expandtabs(4), pairwise_distances.min().item()) 
    print("\t mean".expandtabs(4), pairwise_distances.mean().item())
    print("\t max ".expandtabs(4), pairwise_distances.max().item())
    return pairwise_distances.min() if return_min else out
    
def accuracy_score(y_pred, y_true):
    return (y_pred.argmax(1) == y_true).sum()/y_true.shape[0]

def best(metric_values: list, metric_name: str) -> int:
    r"""
    Returns the best_epoch of the specified metric values.
    
    Args:
        metric_values (list):
            list of values
        metric_name (str):
            name of the metric. 
            
    Returns:
        best_epoch (int):    
            If ``metric_name`` is "mae" or "cosine_embedding_loss", the function returns the epoch 
            corresponding to the minimum of ``metric_values``; otherwise, 
            it returns the epoch corresponding to the maximum of ``metric_values``.
    """
    if metric_name in ["mae", "cosine_embedding_loss"]:
        best_epoch = np.argmin(metric_values)
        mode = "min"
    else:
        best_epoch = np.argmax(metric_values)
        mode = "max"
    return best_epoch

def custom_cosine_embedding_loss(y_pred, y_true):
    r"""
    References:
        [1] Wang, Yanbo, and Muhan Zhang. "Towards Better Evaluation of GNN Expressiveness with BREC Dataset." 
    """
    input1 = y_pred[0::2]
    input2 = y_pred[1::2]
    target = - torch.ones(input1.shape[0], device=device)
    return cosine_embedding_loss(input1, input2, target)


def TSquared(y_pred, y_true):
    r"""
    Compute the T squared statistics from [1]. For more details, refer to the
    BREC datase in src/data/dataset_brec
    
    References:
        [1] Wang, Yanbo, and Muhan Zhang. "Towards Better Evaluation of GNN Expressiveness with BREC Dataset." 
    """
    
    D = y_pred[0::2].T - y_pred[1::2].T
    D_mean = torch.mean(D, dim=1).reshape(-1, 1)
    S = torch.cov(D)
    if torch.linalg.norm(S)<1e-8:
        #prenvent the inversion of a null matrix
        S = S + torch.diag(
            torch.ones(S.shape[0]) * 1e-8
        ).to(device)
    score = D_mean.T @ torch.linalg.lstsq(S, D_mean).solution
    return score