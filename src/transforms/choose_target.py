from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform

@functional_transform('choose_target')
class ChooseTarget(BaseTransform):
    """Replace y with its ``target`` column, i.e., y[:, target].

    Args:
        target (int): column of y to train against.
    """
    
    def __init__(self, target: int):
        self.target = target
        
    def __call__(self, data: Data) -> Data:
        data.y = data.y[:, self.target].unsqueeze(-1)
        return data

    def __repr__(self) -> str:
        out = (
            f'{self.__class__.__name__}'
            +f'(target={self.target})'
        )
        return out