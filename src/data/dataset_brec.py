import networkx as nx
import numpy as np
import os
import random
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import from_networkx
from urllib.request import urlretrieve as download_url
from tqdm import tqdm

class BREC(InMemoryDataset):
    """
    THe raw files are downloaded from <https://github.com/GraphPKU/BREC>.
    
    Given $N-1$ permutations \{p_i\}_{i=1}^{N-1}, we compute for each pair 
    (G, H) a dataset comprising the permutation of G and H as
    
    (G, H, \pi_1(G), \pi_1(H) ..., \pi_{N-1}(G), \pi_{N-1}(H)).
    
    We set the ``train_split`` to be 1/2 of the dataset, the ``val_split`` and
    ``test_split`` to be the 1/4 each. The ``test_split`` is used for the T-squared
    test, while the validation is used to retrieve the best model, i.e., the one
    that performs better in the contrastive loss. For the reliability test, we
    take all permutation of a single graph chosen randomly between G and H in
    the test dataset. 
    
    Args:
        subset (list of ints): which pairs to consider.
        num_permutations (int): how many random permutations of each graph 
            to generate.
            
    Notes:
        the name is expected to be brec_<implemented>, where implemented is
        one among  "basic", "regular", "str", "extension", "cfi", "4vtx",
        "dr". Moreover, the processed files are never saved. Please, be sure
        that ``num_permutations`` is a multiple of 4, and that the order of 
        elements is preserved.
    """
    
    def __init__(
        self, 
        root: str, 
        subset: list = None, 
        num_permutations: int = 32, 
        transform=None, 
        pre_transform=None
    ):
        self.name = os.path.basename(root).split("_")[-1]
        assert num_permutations % 4 ==0, (
            f"Please be ensure that num_permutations is divisible by 4."
            )
        self.num_permutations = num_permutations
        self.subset = subset
        assert self.name in self.implemented, f"BREC_{self.name} not implemented!"
        super().__init__(root, transform, pre_transform)
        self.num_original_graphs = len(self) // num_permutations
        
    @property
    def implemented(self):
        return ["basic", "regular", "str", "extension", "cfi", "4vtx", "dr"]
    @property
    def raw_file_names(self):
        return [self.name+".npy"]

    @property
    def processed_file_names(self):
        return  [f"processed_{self.num_permutations}.pt"]
    
    def download(self):
        if not os.path.exists(self.raw_paths[0]):
            url = os.path.join(
                "https://raw.githubusercontent.com/GraphPKU/BREC/Release/customize/Data/raw", 
                self.raw_file_names[0]
            )
            print(f"Downloading {url}")
            download_url(
                url,
                self.raw_paths[0]
            )  
            
    def process(self):
        # This preprocessing is adapted from the original repo
        data_list = np.load(self.raw_paths[0])
        if self.name in ["basic", "str", "4vtx"]:
            data_list = [data.encode() for data in data_list]
        elif self.name in ["dr"]:
            pass
        elif self.name in ["regular", "cfi"]:
            data_list = [ 
                data for data_pair in data_list for data in data_pair
            ]
        elif self.name in ["extension"]:
            data_list = [
                data.encode() for data_pair in data_list for data in data_pair
            ]
        else:
            assert False, f"BREC {self.name} not implemented!"
        data_list = [
            nx.from_graph6_bytes(data) for data in data_list
        ]
        # Consider only the pairs in subset
        if self.subset is None:
            self.subset = np.arange(len(data_list)//2).tolist()
        idx = [k for p in self.subset for k in [2*p, 2*p+1]]
        data_list = [data_list[i] for i in idx] 
        num_original_graphs = len(data_list)
        # We augment the dataset with permutations of each pair
        print(
            f"Augmenting the original {num_original_graphs} graphs" 
            + f" by {self.num_permutations} permutations."
        )
        augmented_data_list = []
        for i in tqdm(range(0, num_original_graphs, 2), desc=f"BREC {self.name}"):
            G = data_list[i]
            H = data_list[i+1]
            augmented_data_list.extend([G, H])
            for idx in range(1, self.num_permutations):
                # randomly permutations
                relabel_mapping = dict(zip(
                    G.nodes(), 
                    sorted(G.nodes(), key=lambda k: random.random())
                ))
                augmented_data_list.extend(
                    [
                       nx.relabel_nodes(G, relabel_mapping),
                       nx.relabel_nodes(H, relabel_mapping)
                    ]
                )           
        augmented_data_list = [
            from_networkx(data) for data in augmented_data_list
        ]
        if self.pre_filter is not None:
            augmented_data_list = [data for data in augmented_data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            augmented_data_list = [self.pre_transform(data) for data in augmented_data_list]

        self.data_list = augmented_data_list
    
    def __len__(self):
        return len(self.data_list)
    
    
    def __getitem__(self, k):
        return self.data_list[k]
        
    def get_idx_split(self, train_len = .5, val_len = .25):
        num_pairs = self.num_original_graphs // 2
        train_mask = []
        val_mask = []
        test_mask = []        
        for i in range(0, num_pairs):
            train_mask.append(
                 np.arange(
                    2 * i * self.num_permutations,
                    int(2 * (i + train_len) * self.num_permutations) 
                ).tolist()
            )
            val_mask.append(
                 np.arange(
                    int(2 * (i + train_len) * self.num_permutations),
                    int(2 * (i + train_len + val_len) * self.num_permutations)
                ).tolist()
            )
            test_mask.append(
                 np.arange(
                    int(2 * (i + train_len + val_len) * self.num_permutations),
                    int(2 * (i + 1) * self.num_permutations)
                ).tolist()
            )
        return train_mask, val_mask, test_mask