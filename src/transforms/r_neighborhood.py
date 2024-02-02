from collections import Counter
import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_networkx
from typing import Tuple, Union, Iterable, Optional
import warnings

def r_neighborhood(
    G: nx.Graph, 
    r: int,
    edge_index: Optional[Union[Iterable, None]] = None,
    pre_compute: bool = True
) -> Tuple[dict, dict]:
    """
    Compute all paths of length L≤r among distinct neighbors of ``center``.
    
    Args:
        G (networkx.Graph)
        r (int): order of the neighborhood
        edge_index (np.array of dimension (2, #edges)): if provided, it
            returns the index of the corresponding edge_attr, i.e., the position
            of (j, k) in edge_index where j is a node in G and k a neighbor of j
        pre_compute: it precomputes all possible paths. Since most of them are
            permutations of each other, this could possibly lead to a memory overload
            especially when the graphs are dense. The benefit of pre_compute is
            that it simplifies the forward, because everything is already precomputed
    
    Returns:
        paths (dict):
            - key: path length L
            - value: a list containing paths of length L between direct 
            neighbors of center.                
        hop (dict): if pre_compute is True, it is a dict
            - key: path length L
            - value: distance of each node in the path to the center node.
            otherwise it is a dict with keys a pair of nodes (s, t) and value 
            their distance d(s, t)  in the graph
        r_edge_attr_idx (dict):
            - key: path length L
            - value: column of (j, k) in edge_index. 
            
    Dims:
        for each L, paths[L] is of dim. (L+2, num. paths), while r_edge_attr_idx[L] is of
        dimension (L+1, num. paths). Note that num.paths can be also 0.
        If pre_compute is False, the dimension change from from ( , num_paths)
        to (, num_cycles); paths are obtained from permutation of simple cycles.
        In this case, the computation of permutations is done in the forward step.
    """
    cycles = list(nx.simple_cycles(G, length_bound=r+2))
    distances = dict(nx.all_pairs_shortest_path_length(G))
    paths = {}
    if pre_compute:
        hops = {}
    else:
        hops = {(source, target): d for source, target_dict in distances.items() for target, d in target_dict.items()}
    edge_attr_idx = {}
    for L in range(2, r+3):
        # Initializing, useful to have an array of dim (L+1, 0)
        paths[L-2] = np.zeros((L, 0))
        if pre_compute:
            hops[L-2] = np.zeros((L, 0))
        edge_attr_idx[L-2] = np.zeros((L-1, 0))
        # Dividing the simple cycles depending on the length
        # Taking all cyclic permutation via np.roll
        if pre_compute:
            L_long_cycles = [
                np.roll(cycle, current_L).tolist() for cycle in cycles for current_L in range(len(cycle)) if len(cycle)==L 
            ]
        else:
            L_long_cycles = [
                cycle for cycle in cycles if len(cycle)==L 
            ]
        if L_long_cycles:
            # Adding center of neighborhood as last element; useful to compute
            # the index of each edge_attr. Note that paths[L-2] has dim.
            # (num. paths, L+3)
            paths[L-2] =  np.array(
                sorted(L_long_cycles)
            )
            if pre_compute:
                # Storing the distance of each node in the path from the center
                hops[L-2] =np.array([
                    [distances[cycle[0]][target] for target in cycle] for cycle in paths[L-2]
                ]).T
    if edge_index is not None:
        edge_attr_idx_dict = {}
        # Finding index of (src, dst) in edge_index
        for idx, (src, dst) in enumerate(edge_index.transpose()):
            edge_attr_idx_dict[(src, dst)] = idx
        if pre_compute:
            for L, L_paths in paths.items():
                if L_paths.shape[1]>0:
                    edge_attr_idx[L] = np.array(
                        [
                            [
                                edge_attr_idx_dict[src, dst] for src, dst in zip(current_path[:-1], current_path[1:])
                            ] for current_path in L_paths
                        ],
                    ).T
        else:
            edge_attr_idx = edge_attr_idx_dict
    for L in paths.keys():
        if paths[L].shape[1]>0:
            paths[L] = paths[L].T
    return paths, hops, edge_attr_idx

@functional_transform('r_neighborhood')
class rNeighborhood(BaseTransform):
    r'''
    Transform that computes r-neighborhood of each node in the graph. It add to 
    each Data the values
    - loopyN{L}, L-neighborhood with L≤r
    - loopyA{L}, atomic type of each node in L-neighborhood
    - loopyE{L}, edge attribute index of each edge in L-neighborhood
    where L ≤ r is the path length.
    
    Args:
        r (int): maximal length of path among direct neighbors (default: 0).
        pre_compute (bool): if all paths should be precomputed. 
        
    Notes:
        if pre_compute is True, loopyA{L} is not created; instead loopyA (with no 
        dependence on L) will store  pairs of nodes as key and their distance
        in the graph as value.
    '''
    def __init__(self, r: int = 0, pre_compute: bool = True):
        self.r=r
        self.pre_compute = pre_compute
    
    def __call__(self, data: Data) -> Data:
        G = to_networkx(data)
        paths, hops, edge_attr_idx = r_neighborhood(
            G, 
            r = self.r,
            edge_index=data.edge_index.cpu().numpy(),
            pre_compute=self.pre_compute
        )
        if not self.pre_compute:
            data[f"loopyA"] = hops
            data[f"loopyE"] = edge_attr_idx
        for L in paths.keys():
            data[f"loopyN{L}"] = torch.tensor(paths[L], dtype=int)
            if self.pre_compute:
                data[f"loopyE{L}"] = torch.tensor(edge_attr_idx[L], dtype=int)
                data[f"loopyA{L}"] = torch.tensor(hops[L], dtype=int)
        return data
    
    def __repr__(self) -> str:
        out = (
            f'{self.__class__.__name__}'
            + f'(r={self.r})'
        )
        return out
    
def r_stats(dataset, r: int, lazy: bool):
    r"""For each value of length L<=r, it prints the number of graphs that
    have L as maximal path length as well as the min, median, max and total
    number of paths of length L.
    """
    log_text = f'{"L≤r":<6} {"#graphs with maxi-":>20}'
    log_text += f' {"#paths of length L":>30}' if not lazy else f' {"#paths of length L":>30}'
    log_text += f'\n{"":<6} {"mal path length L":>20} {"(min, median, max, total)":>30}'
    print(log_text)
    shapes = [
        [
            data[f"loopyN{L}"].shape[1] for L in range(0, r+1)
        ] for data in dataset
    ]
    stats = np.vstack(
        [np.min(shapes, axis=0), np.median(shapes, axis=0), np.max(shapes, axis=0), np.sum(shapes, axis=0)]
    )
    nonzero_shapes = [
        [
            s>0 for s in shape
        ] for shape in shapes
    ]
    idx_maximal_L = [
        (np.arange(r+1)[row]).max() for row in nonzero_shapes
    ]
    counts = Counter(idx_maximal_L)
    for L, n in counts.items():
        print(f'{str(L):<6} {str(n):>20} {", ".join(map(lambda k: str(int(k)), stats[:, L])):>30}')
    if r not in counts.keys():
        warnings.warn(
            f"No graph with maximal path length {r}."
            "You may consider increasing r."
        )