import networkx as nx
import numpy as np
import os
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import to_undirected
from urllib.request import urlretrieve as download_url



class Synthetic(InMemoryDataset):
    r"""
    Retrieves synthetic datasets as COSPECTRAL10, GRAPH8C, SR16622
    """

    def __init__(self, root, transform=None, pre_transform=None):
        self.name = os.path.basename(root)
        assert self.name in self.implemented, f'Dataset {self.name} not implemented!'
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def implemented(self):
        return ["sr16622", "graph8c", "cospectral10"]
    
    @property
    def raw_file_names(self):
        return [f"{self.name}.g6"]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        filename = os.path.join(
            self.raw_dir, 
            self.name+".g6"
        )
        if not os.path.exists(self.raw_paths[0]):
            if self.name in ["sr16622"]:
                # https://www.maths.gla.ac.uk/~es/srgraphs.php
                url = (
                    "https://raw.githubusercontent.com/gasmichel/"
                    "PathNNs_expressive/main/synthetic/data/SR25/sr16622.g6"
                )
                print(f"Downloading {url}")
                download_url(
                    url,
                    filename
                )  
            elif self.name in ["cospectral10"]:
                print(f"Building dataset from adjacency matrices.")
                A1 = (
                    "0101010100"
                    "1011100000"
                    "0100101001"
                    "1100010100"
                    "0110001001"
                    "1001000011"
                    "0010100110"
                    "1001001010"
                    "0000011101"
                    "0010110010"
                )
                A2 = (
                    "0101001100"
                    "1011100000"
                    "0100110001"
                    "1100010100"
                    "0110001001"
                    "0011000110"
                    "1000100011"
                    "1001010010"
                    "0000011101"
                    "0010101010"
                )
                with open(filename, "wb") as f:
                    for A in [A1, A2]:
                        A = np.array(
                            list(map(int, A))
                        ).reshape(10,10)
                        G = nx.from_numpy_array(A)
                        f.write(nx.to_graph6_bytes(G))        
            else:
                url = os.path.join(
                    "https://users.cecs.anu.edu.au/~bdm/data", 
                    self.name+".g6"
                )
                print(f"Downloading {url}")
                download_url(
                    url, 
                    filename
                )

    def process(self):
        # Read data into huge `Data` list. 
        b=self.processed_paths[0]   
        dataset = nx.read_graph6(
            self.raw_paths[0]
        )
        data_list = []
        for i, datum in enumerate(dataset):
            edge_index = to_undirected(
                torch.tensor(
                    list(datum.edges())
                ).transpose(1, 0)
            )            
            data_list.append(
                Data(
                    edge_index=edge_index,
                    num_nodes=datum.number_of_nodes()
                )
            )
            
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])