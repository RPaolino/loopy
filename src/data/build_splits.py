

import numpy as np
import os
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch_geometric.data import InMemoryDataset
from typing import List, Union
from urllib.request import urlretrieve as download_url
    
def build_splits(
    dataset: Union[InMemoryDataset, List[InMemoryDataset]],
    dataset_name: str,
    num_reps: int
) -> List[tuple]:
    r"""Return the indices of train, val and test datasets.
    
    Args:
        dataset (InMemoryDataset or list of InMemoryDataset): the dataset to 
            train on.
        dataset_name (str): name of the dataset.
        num_reps (int): ``zinc``, ``zinc_subset``, ``qm9`` have only one standard 
            split; hence, this one split is repeated ``num_reps`` times and 
            different seeds are returned. ``exp`` and ``cexp`` have no standard 
            splits; hence, a Stratified ``num_reps``-fold cross validation
            is performed.
            
            
    Returns:
        splits (list of tuples): each tuple has train, val and test indices.
        seeds (list of bools): list of constant if the dataset has different 
            splits; [1, ..., num_reps] if the same split is repeated.
    """
    dataset_name = dataset_name.lower()
    dataset_dir = os.path.join("./datasets", dataset_name)
    if dataset_name in ["zinc_subset", 
                        "zinc"]:
        lengths = [len(d) for d in dataset]
        idx = np.cumsum(lengths)
        # Repeating the mask
        train_idx = [
            list(np.arange(idx[0])) for _ in range(num_reps)
        ]
        val_idx = [
            list(np.arange(idx[0], idx[1])) for _ in range(num_reps)
        ]
        test_idx = [
            list(np.arange(idx[1], idx[2])) for _ in range(num_reps)
        ]
        seeds = np.arange(1, num_reps+1)
    elif dataset_name.startswith("qm9"):
        idx = [int(percent*len(dataset)) for  percent in [.8, .1, .1]]
        idx = np.cumsum(idx)
        train_idx = [
            list(np.arange(idx[0])) for _ in range(num_reps)
        ]
        val_idx = [
            list(np.arange(idx[0], idx[1])) for _ in range(num_reps)
        ]
        test_idx = [
            list(np.arange(idx[1], idx[2])) for _ in range(num_reps)
        ]
        seeds = np.arange(1, num_reps+1)   
    elif dataset_name.startswith("brec"):
        train_idx, val_idx, test_idx = dataset.get_idx_split()
        seeds = [1000]*len(train_idx)     
    elif dataset_name in ["exp_iso", "cospectral10", "graph8c", "sr16622"]:
        # Since there is no training, we initialize the train split with empty
        # lists.
        train_idx = [[] for _ in range(num_reps)]
        # Repeating the mask
        val_idx = [
            list(np.arange(len(dataset))) for _ in range(num_reps)
        ]
        test_idx = [[] for _ in range(num_reps)]
        seeds = np.arange(1, num_reps+1)
    elif dataset_name in ["csl"]:
        base_url = ("https://raw.githubusercontent.com/"
                    "gasmichel/PathNNs_expressive/main/synthetic/data/CSL/CSL_")
        split_names = ['train', 'val', 'test']
        idx = {}
        os.makedirs(
            os.path.join(dataset_dir, "raw"), 
            exist_ok=True
        )
        for split in split_names:
            filename = f"{split}.index"
            path = os.path.join(
                dataset_dir, 
                "raw", 
                filename
            )
            url = base_url + filename
            if not os.path.exists(path):
                print(f"Downloading {url}")
                download_url(
                    url, 
                    path
                )
            with open(path, 'r') as f:
                idx[split]  = [
                    [
                        int(x) for x in line.split(",")
                    ]  for line in f.read()[:-1].split('\n')
                ]
        train_idx = idx["train"]
        val_idx = idx["val"]
        test_idx = idx["test"]
        seeds = [1000]*len(train_idx)
    elif dataset_name in  ["exp", "cexp"]:
        # Initializing Stratified k-fold cross validation      
        kf = StratifiedKFold(n_splits=num_reps)
        y = dataset._data.y.cpu().numpy()
        # Creating test idx
        train_idx, test_idx = train_test_split(
            range(len(dataset)), 
            test_size=.2,
            stratify=y
        )
        # Kfold from train_idx
        out = [
            (train, val, test_idx) for train, val in kf.split(train_idx, y[train_idx])
        ]
        return out, [1000]*num_reps
    elif dataset_name.startswith("subgraphcount"):
        train_idx, val_idx, test_idx = dataset.get_idx_split()
        # Repeating the mask
        train_idx = [
            train_idx[0] for _ in range(num_reps)
        ]
        val_idx = [
            val_idx[0] for _ in range(num_reps)
        ]
        test_idx = [
            test_idx[0] for _ in range(num_reps)
        ]
        seeds = np.arange(1, num_reps+1)
    else:
        assert False, f'No splits for {dataset_name}'
    out = list(zip(train_idx, val_idx, test_idx))
    return out, seeds
