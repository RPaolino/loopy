import torch
from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform

@functional_transform("constant_attribute")
class ConstantAttribute(BaseTransform):
    """Add a constant ``name_attr`` attribute to each Data. 

        Args:
            name_attr (str):
                name of the attribute to add.
            value (float, optional): 
                constant value of the attribute. Defaults to 1.
            dtype (optional): 
                dtype of the Tensor corresponding to ``name_attr``. 
                Defaults to torch.float32.
    """
    
    def __init__(self, name_attr: str, value: float = 1, dtype=int):
        self.name_attr = name_attr
        self.value = value
        self.dtype = dtype
        
    def __call__(self, data: Data) -> Data:
        if  self.name_attr == "edge_attr":
            data.edge_attr = torch.ones(
                (data.edge_index.shape[1], 1),
                dtype=self.dtype
            )*self.value
        elif  self.name_attr == "y":
            data.y= torch.ones(
                (1, 1),
                dtype=self.dtype
            )*self.value
        elif  self.name_attr == "x":
            data.x= torch.ones(
                (data.num_nodes, 1),
                dtype=self.dtype
            )*self.value
        else:
            assert False, f"Cannot add {self.name_attr}, not implemented."
        return data

    def __repr__(self) -> str:
        out = (
            f'{self.__class__.__name__}'
            +f'(name_attr={self.name_attr}, value={self.value}, dtype={str(self.dtype)})'
        )
        return out
