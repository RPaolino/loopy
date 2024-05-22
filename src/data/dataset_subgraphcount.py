import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
import tqdm

class Subgraphcount(InMemoryDataset):
    r"""
    From https://github.com/subgraph23/homomorphism-expressivity
    Homomorphism of the following motifs:
        - chordal4: [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3]],
        - boat: [[0, 1], [0, 2], [1, 2], [1, 3], [2, 4], [3, 4], [3, 5], [4, 5]],
        - chordal6: [[0, 1], [0, 2], [1, 2], [1, 3], [1, 4], [2, 4], [2, 5], [3, 4], [4, 5]],
    Subgraph isomorphisms of the following motifs:
        - cycle3: [[0, 1], [1, 2], [2, 0]], 
        - cycle4: [[0, 1], [1, 2], [2, 3], [3, 0]], 
        - cycle5: [[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]], 
        - cycle6: [[0,1], [1, 2], [2, 0], [3, 4], [4, 5], [5, 0]], 
        - chordal4: [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3]], 
        - chordal5: [[0, 1], [1, 2], [2, 3], [3, 4], [4, 0], [0, 2], [0, 3]]
    """
    def __init__(self, root: str, **kwargs):
        super().__init__(root=root.split("_")[0], **kwargs)
        self.data, self.slices = torch.load(
            self.processed_paths[0]
        )

    @property
    def raw_file_names(self):
        return ["graph.npy", "counts.csv"]

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def process(self):

        graph_list, split = np.load(self.raw_paths[0], allow_pickle=True)
        y = np.loadtxt(self.raw_paths[1], delimiter='\t', skiprows=2, dtype=int)
        data_list = []
        for i in tqdm.trange(len(graph_list), desc="Subgraphcount"):
            A = graph_list[i]
            # Edge index
            E = A.nonzero()
            edge_index = torch.Tensor(
                np.vstack(
                    (E[0], E[1])
                )
            ).type(torch.int64)
            # Append Data object
            data_list.append(
                Data(
                    edge_index=edge_index,
                    num_nodes=A.shape[0],
                    y=torch.tensor(y[i]/y[split[()]["train"]].std(0)).unsqueeze(0)
                )
            )
        # Apply pre_filter
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        # Apply pre_transform
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        # Save the data
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self):
        (_, split) = np.load(self.raw_paths[0], allow_pickle=True)
        train_idx = [split[()]['train'].tolist()]
        val_idx = [split[()]['val'].tolist()]
        test_idx = [split[()]['test'].tolist()]
        return train_idx, val_idx, test_idx