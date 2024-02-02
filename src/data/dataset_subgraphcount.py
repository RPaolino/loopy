import numpy as np
import os
from scipy.io import loadmat
import torch
from torch_geometric.data import Data, InMemoryDataset
import tqdm
from urllib.request import urlretrieve as download_url

class Subgraphcount(InMemoryDataset):
    
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(
            root.split("_")[0], 
            transform, 
            pre_transform
        )
        self.data, self.slices = torch.load(
            self.processed_paths[0]
        )

    @property
    def raw_file_names(self):
        return ["randomgraph.mat"]

    @property
    def processed_file_names(self):
        return [f"data.pt"]

    def download(self):
        if not os.path.exists(self.raw_paths[0]):
            download_url(
                url=(
                    "https://raw.githubusercontent.com/LingxiaoShawn/GNNAsKernel"
                    +"/main/data/subgraphcount/raw/randomgraph.mat"
                ),
                filename=self.raw_paths[0]
            )

    def process(self):
        # Read data into huge `Data` list. 
        b=self.processed_paths[0]       
        a=loadmat(self.raw_paths[0]) #'subgraphcount/randomgraph.mat')
        # list of adjacency matrix
        A_list=a['A'][0]
        # list of output
        Y=a['F']
        # Initialize list of Data object
        data_list = []
        for i in tqdm.trange(len(A_list), desc="Subgraphcount"):
            A = A_list[i]
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
                    num_nodes=A.shape[0]
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
        a = loadmat(self.raw_paths[0])
        train_idx = a['train_idx']
        val_idx = a['val_idx']
        test_idx = a['test_idx']
        return train_idx, val_idx, test_idx