from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform

@functional_transform('delete_attribute')
class DeleteAttribute(BaseTransform):
    """Delete the atribute whose name is name_attr from each Data.

        Args:
            name_attr (str): name of the attribute to remove
    """
    
    def __init__(self, name_attr: str):
        self.name_attr = name_attr
        
    def __call__(self, data: Data) -> Data:
        delattr(data, self.name_attr)
        return data
    
    def __repr__(self) -> str:
        out = (
            f'{self.__class__.__name__}'
            +f'(name_attr={self.name_attr})'
        )
        return out