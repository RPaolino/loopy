import os
import pickle
import torch
from torch_geometric.data import Data, InMemoryDataset
import tqdm
from urllib.request import urlretrieve as download_url

class PlanarSATPairs(InMemoryDataset):
    
    def __init__(
        self, 
        root: str, 
        transform=None, 
        pre_transform=None, 
        pre_filter=None
    ):
        """Adatpted from the github repo <https://github.com/ralphabb/GNN-RNI/tree/main>
        of [1].

        [1] R. Abboud, I.I. Ceylan, M. Grohe, & T. Lukasiewicz (2021). 
            The Surprising Power of Graph Neural Networks with Random Node 
            Initialization. In Proceedings of the Thirtieth International Joint 
            Conference on Artifical Intelligence (IJCAI).
        """
        super().__init__(
            root, 
            transform,
            pre_transform, 
            pre_filter
        )
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def name(self):
        return os.path.basename(self.root)
    
    @property
    def raw_file_names(self):
        return ["graphsat.pkl"]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        if not os.path.exists(self.raw_paths[0]):
            url = os.path.join(
                "https://raw.githubusercontent.com/ralphabb/GNN-RNI/main/Data",
                self.name.upper(),
                "raw",
                "GRAPHSAT.pkl"
            )
            print(f"Downloading {url}")
            download_url(
                url, 
                self.raw_paths[0]
            )

    def process(self):
        # Read data into huge `Data` list.
        filename = self.raw_paths[0]
        with open(filename, "rb") as f:
            data = pickle.load(f)    
        # The 'data' object was created by an older version of PyG. This function
        # helps avoiding the downgrade of PyG to older versions.
        # https://github.com/pyg-team/pytorch_geometric/discussions/7241
        data_list = []
        for d in tqdm.tqdm(data, desc="Backward compatibility"):
            data_list.append(
                Data.from_dict(d.__dict__)
            )
        
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])