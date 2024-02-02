import numpy as np
import os
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.transforms import (
    BaseTransform,
    Compose
)
import tqdm
from typing import List, Union, Iterable

from .custom_collate import custom_collate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def freeze(
    dataset: Union[InMemoryDataset, List[InMemoryDataset]],
    dataset_name: str,
    r: int,
    transform: Union[BaseTransform, Compose],
    task: str
) -> InMemoryDataset:
    r"""If ``transform`` is passed to an InMemoryDataset instance, the transform 
    is applied every time a Data element is retrieved (slow!). 
    If ``pre_transform`` is passed to an InMemoryDataset instance, the data is 
    first transformed, collated in a huge Data object and then stored on the disk.
    This means that every time the ``pre_transform`` changes, one has to manually 
    delete the processed files (tedious!). Moreover, every time a Data element 
    is called, one has to compute the specific indices corresponding to that 
    particular element in the huge collated Data (slow!).
    
    On the contraty, this function applies transform to each Data in dataset 
    and store them in memory, avoiding to store them on the disk and to compute 
    the transform at each call (efficient!).
    
    Args:
        dataset (InMemoryDataset of list of InMemoryDataset): dataset to train on.
        dataset_name (str): name of ``dataset``.
        r (int): order of loopy neighborhood.
        transform (BaseTransform or Compose of BaseTransform): the transform(s) 
        to apply to each element of ``dataset``.
        task (str): task corresponding to the dataset, e.g., "graph_regression".
    """
    dataset_name = dataset_name.lower()
    dataset_dir = os.path.join("./datasets", dataset_name)
    if dataset_name in ["zinc_subset", 
                        "zinc"]:
        frozen_dataset = Frozen(
            dataset=([d for d in dataset[0]]
                    + [d for d in dataset[1]]
                    + [d for d in dataset[2]]),
            name=dataset_name,
            root=dataset_dir,
            task=task,
            r=r,
            transform=transform
        )
    else:
        frozen_dataset = Frozen(
            dataset=dataset,
            name=dataset_name, 
            root=dataset_dir,
            r=r,
            task=task,
            transform=transform
        )
    return frozen_dataset

class Frozen(InMemoryDataset):
    r"""InMemoryDataset where the Data elements are stored in a list."""
    
    def __init__(
        self,
        dataset: InMemoryDataset,
        name: str,
        root: str,
        r: int,
        task: str,
        transform: Union[BaseTransform, Compose]
    ):
        
        super().__init__()
        self.task = task
        self.name = name
        self.transform = transform
        current_data_folder = os.path.join(root, f"r_{r}")
        # Showing the progress bar
        progress = tqdm.trange(
            len(dataset), 
            desc="\tFreezing".expandtabs(4)
        )
        self._data_list = [
            transform(dataset[k].clone()).to(device)  for k in progress
        ]
        # Collating all elements of the dataset
        self._data = custom_collate(
            self._data_list
        )
        # Computing num. classes
        try:
            self._num_classes = self._data_list[0].y.shape[1]
        except:
            self._num_classes = dataset.num_classes
        # Computing number of node features
        self._num_features = self._data_list[0].x.shape[1]
        # Computing number of edge features
        self._num_edge_features=self._data_list[0].num_edge_features
            
    def __len__(self) -> int:
        return len(self._data_list)

    def __getitem__(self, idx: Union[int, slice, Iterable]) -> list:
        if isinstance(idx, int):
            out = self._data_list[idx]
        elif isinstance(idx, slice):
            start = 0 if idx.start is None else idx.start
            stop = len(self._data_list) if idx.stop is None else idx.stop
            step = 1 if idx.step is None else  idx.step
            out = [
                self._data_list[current_idx] for current_idx in np.arange(start, stop, step)
            ]
        else:
            out = [
                self._data_list[current_idx] for current_idx in idx
            ]
        return out
    
    @property
    def num_classes(self) -> int:
        return self._num_classes
    
    @property
    def num_node_features(self) -> int:
        return self._num_features
    
    @property
    def num_edge_features(self) -> int:
        return self._num_edge_features
        
    def __repr__(self):
        out = f'Frozen{self.name}' + f'(len={len(self._data_list)})'
        return out