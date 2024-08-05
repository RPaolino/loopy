import numpy as np
import os
from torch_geometric.data import InMemoryDataset
from torch_geometric.datasets import (
    GNNBenchmarkDataset,
    ZINC,
    QM9
)
from typing import List, Union

from .dataset_brec import BREC
from .dataset_subgraphcount import Subgraphcount
from .dataset_planar_sat_pairs import  PlanarSATPairs
from .dataset_synthetic import Synthetic
from .dataset_peptides import Peptides
   
    
class GNNBenchmark(GNNBenchmarkDataset):
    r"""
    Overwriting methods to have non-capital ``raw_dir`` and ``processed_dir`` names.
    """
    @property
    def raw_dir(self) -> str:
        return super().raw_dir.lower()

    @property
    def processed_dir(self) -> str:
        return super().processed_dir.lower()

def build_dataset(
    dataset_name: str,
    r: int,
    num_reps: int
) -> Union[InMemoryDataset, List[InMemoryDataset]]:
    r"""Load the plain dataset.
    
    Args:
        dataset_name (str): name of the dataset to consider.
        r (int): order of neighborhood. Useful for BREC datasets, since we load
            the indistinguished pairs by (r-1) from 
            ``equivalent/<dataset_name>_{r-1}lWL.csv``
        num_reps (int): Useful for BREC datasets; it specifies the number of 
            permutations of each graph.
            
    Returns:
        dataset (InMemoryDataset or list of InMemoryDataset): for ``zinc`` and 
        ``zinc_subset``, this function returns a list comprising train, val 
        and test datasets.
    """
    dataset_name=dataset_name.lower()
    dataset_dir = os.path.join("./datasets", dataset_name)
    if dataset_name in ["zinc_subset", "zinc"]:
        dataset = [
            ZINC(
                root=dataset_dir.replace("_subset", ""),
                subset=True if "subset" in dataset_name else False,
                split=split,
            ) for split in ["train", "val", "test"]
        ]
    elif dataset_name == "csl":
        dataset = GNNBenchmark(
            name="CSL",
            root=dataset_dir.replace(dataset_name, "")
        )
    elif dataset_name in ["exp", "cexp", "exp_iso"]:
        dataset = PlanarSATPairs(
            root=dataset_dir.replace("_iso", "")
        )
    elif dataset_name.startswith("brec"):
        subset = None
        filepath = f"equivalent/{dataset_name}_{r-1}lWL.csv"
        if os.path.exists(filepath):
            subset = np.genfromtxt(
                filepath, 
                delimiter='\n',
                dtype=int
            ).flatten()
            print(f"Uploading {len(subset)} pairs indices from {filepath}.")
        dataset = BREC(
            root=dataset_dir,
            subset=subset,
            num_permutations=num_reps
        )
    elif dataset_name.startswith("subgraphcount"):
        dataset = Subgraphcount(
            root=dataset_dir
        )
    elif dataset_name in ["graph8c", "cospectral10", "sr16622"]:
        if dataset_name=="graph8c":
            idx = np.unique(
                np.genfromtxt(
                    './equivalent/graph8c_1WL.csv', 
                    delimiter=',',
                    dtype=int
                ).flatten()
            )
            idx = {
                k: r for r, k in enumerate(idx)
            }
            dataset = Synthetic(
                root=dataset_dir
            )[list(idx.keys())]
        else:
            dataset = Synthetic(
                root=dataset_dir
            )
    elif dataset_name.startswith("qm9"):
        dataset = QM9(
            root=dataset_dir.split("_")[0]
        ).shuffle()
    elif dataset_name.startswith("peptides"):
        dataset = Peptides(
            root=dataset_dir
        )
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not implemented.")
    return dataset
