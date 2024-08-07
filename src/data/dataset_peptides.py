import hashlib
import os.path as osp
import pickle
import shutil

import pandas as pd
import torch
from ogb.utils.mol import smiles2graph
from ogb.utils.url import decide_download
from torch_geometric.data import Data, InMemoryDataset, download_url
from tqdm import tqdm

class Peptides(InMemoryDataset):
    def __init__(self, root, smiles2graph=smiles2graph,
                 transform=None, pre_transform=None):
        """
        PyG dataset of 15,535 peptides represented as their molecular graph
        (SMILES) with 10-way multi-task binary classification of their
        func classes (peptides_func) or 11 regression targets 
        derived from the peptide's 3D structure (peptides_struct).

        The 10 classes represent the following func classes (in order):
            ['antifungal', 'cell_cell_communication', 'anticancer',
            'drug_delivery_vehicle', 'antimicrobial', 'antiviral',
            'antihypertensive', 'antibacterial', 'antiparasitic', 'toxic']

        The 11 regression targets were precomputed from molecule XYZ:
            Inertia_mass_[a-c]: The principal component of the inertia of the
                mass, with some normalizations. Sorted
            Inertia_valence_[a-c]: The principal component of the inertia of the
                Hydrogen atoms. This is basically a measure of the 3D
                distribution of hydrogens. Sorted
            length_[a-c]: The length around the 3 main geometric axis of
                the 3D objects (without considering atom types). Sorted
            Spherocity: SpherocityIndex descriptor computed by
                rdkit.Chem.rdMolDescriptors.CalcSpherocityIndex
            Plane_best_fit: Plane of best fit (PBF) descriptor computed by
                rdkit.Chem.rdMolDescriptors.CalcPBF

        Args:
            root (string): Root directory where the dataset should be saved.
            smiles2graph (callable): A callable function that converts a SMILES
                string into a graph object. We use the OGB featurization.
                * The default smiles2graph requires rdkit to be installed *
        """

        self.root = root
        self.smiles2graph = smiles2graph
        self.basename = osp.basename(root)
        if "func"  in self.basename:
            self.url = 'https://www.dropbox.com/s/ol2v01usvaxbsr8/peptide_multi_class_dataset.csv.gz?dl=1'
            self.version = '701eb743e899f4d793f0e13c8fa5a1b4'  # MD5 hash of the intended dataset file
            self.url_stratified_split = 'https://www.dropbox.com/s/j4zcnx2eipuo0xz/splits_random_stratified_peptide.pickle?dl=1'
            self.md5sum_stratified_split = '5a0114bdadc80b94fc7ae974f13ef061'
        elif "struct" in self.basename:
            self.url = 'https://www.dropbox.com/s/0d4aalmq4b4e2nh/peptide_structure_normalized_dataset.csv.gz?dl=1'
            self.version = 'c240c1c15466b5c907c63e180fa8aa89'  # MD5 hash of the intended dataset file
            self.url_stratified_split = 'https://www.dropbox.com/s/9dfifzft1hqgow6/splits_random_stratified_peptide_structure.pickle?dl=1'
            self.md5sum_stratified_split = '5a0114bdadc80b94fc7ae974f13ef061'
        else:
            assert False, f"{self.basename} not implemented!"
       
        # Check version and update if necessary.
        release_tag = osp.join(self.root, self.version)
        if osp.isdir(self.root) and (not osp.exists(release_tag)):
            print(f"{self.__class__.__name__} has been updated.")
            if input("Will you update the dataset now? (y/N)\n").lower() == 'y':
                shutil.rmtree(self.root)

        super().__init__(self.root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        if "func"  in self.basename:
            return 'peptide_multi_class_dataset.csv.gz'
        elif "struct" in self.basename:
            return f'peptide_structure_normalized_dataset.csv.gz'

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def _md5sum(self, path):
        hash_md5 = hashlib.md5()
        with open(path, 'rb') as f:
            buffer = f.read()
            hash_md5.update(buffer)
        return hash_md5.hexdigest()

    def download(self):
        if decide_download(self.url):
            path = download_url(self.url, self.raw_dir)
            # Save to disk the MD5 hash of the downloaded file.
            hash = self._md5sum(path)
            if hash != self.version:
                raise ValueError("Unexpected MD5 hash of the downloaded file")
            open(osp.join(self.root, hash), 'w').close()
            # Download train/val/test splits.
            path_split1 = download_url(self.url_stratified_split, self.root)
            assert self._md5sum(path_split1) == self.md5sum_stratified_split
        else:
            print('Stop download.')
            exit(-1)

    def process(self):
        data_df = pd.read_csv(self.raw_paths[0])
        smiles_list = data_df['smiles']

        print('Converting SMILES strings into graphs...')
        data_list = []
        for i in tqdm(range(len(smiles_list))):
            data = Data()

            smiles = smiles_list[i]
            graph = self.smiles2graph(smiles)

            assert (len(graph['edge_feat']) == graph['edge_index'].shape[1])
            assert (len(graph['node_feat']) == graph['num_nodes'])

            data.__num_nodes__ = int(graph['num_nodes'])
            data.edge_index = torch.from_numpy(graph['edge_index']).to(
                torch.int64)
            data.edge_attr = torch.from_numpy(graph['edge_feat']).to(
                torch.int64)
            data.x = torch.from_numpy(graph['node_feat']).to(torch.int64)
            if "func"  in self.basename:
                data.y = torch.Tensor([eval(data_df['labels'].iloc[i])])
            elif "struct" in self.basename:
                target_names = ['Inertia_mass_a', 'Inertia_mass_b', 'Inertia_mass_c',
                    'Inertia_valence_a', 'Inertia_valence_b',
                    'Inertia_valence_c', 'length_a', 'length_b', 'length_c',
                    'Spherocity', 'Plane_best_fit']
                # Assert zero mean and unit standard deviation.
                assert all(abs(data_df.loc[:, target_names].mean(axis=0)) < 1e-10)
                assert all(abs(data_df.loc[:, target_names].std(axis=0) - 1.) < 1e-10)
                data.y = torch.Tensor([data_df.iloc[i][target_names]])
            data_list.append(data)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)

        print('Saving...')
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self):
        """ Get dataset splits.

        Returns:
            Dict with 'train', 'val', 'test', splits indices.
        """
        if "func"  in self.basename:
            split_file = osp.join(self.root,
                                "splits_random_stratified_peptide.pickle")
        elif "struct" in self.basename:
            split_file = osp.join(self.root,
                        "splits_random_stratified_peptide_structure.pickle")
        with open(split_file, 'rb') as f:
            splits = pickle.load(f)
        return splits["train"].tolist(), splits["val"].tolist(), splits["test"].tolist()
