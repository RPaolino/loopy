import torch
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform



@functional_transform('euclidean_dist_to_edge_attr')
class EuclideanDistToEdgeAttr(BaseTransform):
    """Each node in QM9 is equipped with a ``pos`` attribute. For each edge (i, j)
        we use as edge_attr the euclidean distance between pos[i] and pos[j].
    """      
    
    def __call__(self, data):
        (row, col) = data.edge_index
        dist = torch.norm(
            data.pos[col] - data.pos[row], 
            p=2, 
            dim=-1
        ).unsqueeze(-1)
        dist = dist/dist.max()
        data.edge_attr = torch.cat(
            [
                data.edge_attr,
                dist
            ],
            dim=-1
        )
        return data