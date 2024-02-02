import numpy as np
import torch
from torch_geometric.data import Data

from ..custom_collate import custom_collate
from ...transforms.r_neighborhood import rNeighborhood

def test_star():
    edge_indices = {
        "4star": [(0, 1), (0, 2), (0, 3), (0, 4),
                 (1, 0), (2, 0), (3, 0), (4, 0)],
        "triangle": [(0, 1), (1, 2), (2, 0),
                     (1, 0), (2, 1), (0, 2)],
        "4cycle": [(0, 1), (1, 2), (2, 3), (3, 0),
                   (1, 0), (2, 1), (3, 2), (0, 3)]
    }
    dataset = [
        Data(
            edge_index=torch.LongTensor(edge_index).t().contiguous(), 
            num_nodes=np.max(edge_index)+1,
            x=torch.ones(np.max(edge_index)+1)
        ) for _, edge_index in edge_indices.items()
    ]
    collated = custom_collate(dataset)
    
    assert collated.num_nodes==(5+3+4), (
        f"Num. nodes of collated list is wrong!"
        f"Expected 11, got {collated.num_nodes}"
    )
    
    expected_batch=torch.LongTensor([0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2])
    assert torch.all(collated.batch==expected_batch), f"Batch of collated list is wrong!"
    
    expected_edge_index = torch.LongTensor([
        (0, 1), (0, 2), (0, 3), (0, 4),
        (1, 0), (2, 0), (3, 0), (4, 0),
        (0+5, 1+5), (1+5, 2+5), (2+5, 0+5),
        (1+5, 0+5), (2+5, 1+5), (0+5, 2+5),
        (0+5+3, 1+5+3), (1+5+3, 2+5+3), (2+5+3, 3+5+3), (3+5+3, 0+5+3),
        (1+5+3, 0+5+3), (2+5+3, 1+5+3), (3+5+3, 2+5+3), (0+5+3, 3+5+3)
    ]).t()
    assert torch.all(collated.edge_index==expected_edge_index), f"Edge index of collated list is wrong!"
    
    r = 2
    transform = rNeighborhood(r=r)
    transformed_dataset = [
        transform(data) for data in dataset
    ]
    collated_transformed = custom_collate(transformed_dataset)
    transformed_collated = transform(collated)
    for L in range(r+1):
        for attrib in ["N", "A", "E"]:
            attrib1 = collated_transformed[f"loopy{attrib}{L}"].t().tolist()
            attrib2 = transformed_collated[f"loopy{attrib}{L}"].t().tolist()
            assert sorted(attrib1) == sorted(attrib2), f"loopy{attrib}{L} wrong"