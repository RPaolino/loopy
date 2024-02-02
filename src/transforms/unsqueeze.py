from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform

@functional_transform('unsqueeze')
class Unsqueeze(BaseTransform):  
    """Add a last dimension to ``name_attr`` attribute.

        Args:
            name_attr (str): name of the attribute to unsqueeze.
    """    
    def __init__(self, name_attr : str):
        self.name_attr = name_attr
        
    def __call__(self, data):
        data[self.name_attr] = data[self.name_attr].unsqueeze(-1)
        return data
    
    def __repr__(self):
        return f'{self.__class__.__name__}(name_attr={self.name_attr})'