import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_dense_adj, to_networkx

@functional_transform('count_subgraphs')
class CountSubgraphs(BaseTransform):
    r"""
        It replace (or creates) the ``y`` attribute of each Data with the count 
        of cycles.
        
        Args:
            subgraph (int): length of the cycles to count.
        """
    
    def __init__(self, subgraph: int):
        
        implemented = list(range(3, 7))
        assert subgraph in implemented, (f"Count of {subgraph}cycle not implemented! "
            + f"Possible choices {', '.join(map(str, implemented))}.")
        self.subgraph = subgraph
        
    def __call__(self, data: Data) -> Data:
        A = to_dense_adj(data.edge_index)[0]
        num_nodes = A.shape[0]
        deg = A.sum(0).flatten()
        if self.subgraph == 3:
            A3 = torch.linalg.matrix_power(A, 3)
            count = (A3.trace() / 6)
        elif self.subgraph == 4:
            A2 = torch.linalg.matrix_power(A, 2)
            A4 = torch.linalg.matrix_power(A2, 2)
            # from  https://barumpark.com/blog/2018/Number-of-Four-cycles-in-Networks/
            # (A^2)_{i, i} = \sum_k A_{i, k} A_{k, i}
            # undirected -> A_{i, k} = A_{k, i}
            #              = \sum_k (A_{i, k})^2
            # A_{i, k} \in {0, 1} -> (A_{i, k})^2 = A_{i, k}
            #              = \sum_k  A_{i, k}
            #              = deg_{i}
            # \sum_{i} deg_{i} = trace(A^2) = 2 * #edges
            # --
            # \sum_{i} deg_{i}^2 = \sum_i (\sum_j A_{i, j})^2
            #                    = \sum_i (\sum_j A_{i, j}) (\sum_k A_{i, k})
            #                    = \sum_j \sum_k  \sum_i A_{i, j} A_{i, k}
            # undirected -> A_{i, j} = A_{j, i}
            #                    = \sum_j \sum_k  \sum_i A_{j, i} A_{i, k}
            #                    = \sum_j \sum_k  (A^2)_{j, k}
            #                    = (A^2).sum()
            count = int(
                (
                    A4.trace() # Num. of walks of length 4
                    + A2.trace() # Sum of degrees
                    - 2 * A2.sum() # Sum of squares of degrees
                ) / 8
            )
        elif self.subgraph in [5, 6]:
            G = to_networkx(data)
            cycles = list(nx.simple_cycles(G, length_bound=self.subgraph))
            count = np.sum([
                len(c)==self.subgraph for c in cycles
            ])
            count = np.sum(count)
        data.y = torch.ones((1,1), dtype=torch.float32) * count
        return data

    def __repr__(self) -> str:
        out = (
            f'{self.__class__.__name__}'
            +f'(subgraph={self.subgraph}cycle)'
        )
        return out    
    