import numpy as np
from torch.utils.data import DataLoader
from torch_geometric.data import InMemoryDataset
from typing import List

from .custom_collate import custom_collate
from ..utils import seed_all

def build_loader(
    datasets: List[InMemoryDataset],
    batch_size: int,
    shuffle: List[bool] = [True, False, False],
    seed: int = 1000
) -> List[DataLoader]:
    r"""Build dataloader with `custom_collate` function.
    
    Args:
        datasets (list of InMemoryDataset): list of train, val and test dataset.
        batch_size (int): dimension of batches.
        shuffle (list of booleans): booleans indicating for each dataset if the
            corresponding dataloader should be shuffled. Useful for BREC, 
            where the train_dataset should not be shuffled to not lose the
            correspondence between pairs.
            
    Return:
        loaders (list of DataLoader): list of train, val, test data loaders.
    
    Notes: worker_init_fn for reproducible results
        https://github.com/pytorch/pytorch/issues/7068
    """
    loaders = [
        DataLoader(
            datasets[split], 
            batch_size=batch_size, 
            shuffle=shuffle[split],
            collate_fn=custom_collate,
            worker_init_fn=lambda worker_id: seed_all(seed)      
        ) if (datasets[split] is not None) else None for split in range(len(datasets))
    ]
    return loaders