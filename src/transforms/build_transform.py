from torch_geometric.transforms import Compose
from typing import Union

from .choose_target import ChooseTarget
from .constant_attribute import ConstantAttribute
from .count_subgraphs import CountSubgraphs
from .delete_attribute import DeleteAttribute
from .euclidean_dist_to_edge_attr import EuclideanDistToEdgeAttr
from .r_neighborhood import rNeighborhood
from .to_original_units import ToOriginalUnits
from .unsqueeze import Unsqueeze

def build_transform(
    dataset_name : str,
    r : int,
    target : Union[int, str],
    lazy: bool
) -> Compose:
    r"""
    Built an object `torch_geometric.transforms.Compose` with the specified list of transforms to apply.
    Args
    --
        dataset_name (str): name of the dataset.
        r (int): the specified max path length.
        target (int or str): for qm9, it is an  int specifying which column to regress against;
            for subgraphcount, it is a str specifying which substructure to count
        lazy (bool): if False, all paths are computed, otherwise, they are computed
            in the forward step.
    """
    dataset_name = dataset_name.lower()
    transforms = []
    if (dataset_name in ["csl", "cospectral10", "graph8c", "sr16622"]
        or dataset_name.startswith("subgraphcount")
        or dataset_name.startswith("brec")):
        transforms.append(
            ConstantAttribute("x")
        )
    if dataset_name.startswith("subgraphcount"):
        transforms.append(
            CountSubgraphs(subgraph=target)
        )
    if dataset_name.startswith("qm9"):
        transforms.extend(
            [
                DeleteAttribute("smiles"),
                DeleteAttribute("name"),
                ToOriginalUnits(),
                EuclideanDistToEdgeAttr(),
                ChooseTarget(target=target)
            ]
        )
    if dataset_name in ["zinc", "zinc_subset"]:
        transforms.extend(
            [
                Unsqueeze(name_attr="y"),
                Unsqueeze(name_attr="edge_attr")
            ]
        )
    if (dataset_name in ["csl", 
                        "exp", 
                        "exp_iso", 
                        "cexp",
                        "cospectral10", 
                        "graph8c", 
                        "sr16622"]
        or dataset_name.startswith("subgraphcount")
        or dataset_name.startswith("brec")):
        transforms.append(
            ConstantAttribute("edge_attr")
        )
    if (dataset_name in ["cospectral10", "graph8c", "sr16622"]
        or dataset_name.startswith("brec")):
        transforms.append(
            ConstantAttribute("y")
        )
    transforms.append(
        rNeighborhood(r=r, pre_compute=not lazy)
    )
    # Compose all transforms and return
    return Compose(transforms)