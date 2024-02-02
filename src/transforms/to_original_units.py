import torch
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform



@functional_transform('to_original_units')
class ToOriginalUnits(BaseTransform):
    """The units provided by PyG QM9 are not consistent with their original units.
        We do unit conversion in order to compare with previous work.
    """
    def __init__(self) -> None:
        super().__init__()
        HAR2EV = 27.2113825435
        KCALMOL2EV = 0.04336414
        self.conversion = torch.tensor([
            1., 1., HAR2EV, HAR2EV, HAR2EV, 1., HAR2EV, HAR2EV, HAR2EV, HAR2EV, HAR2EV,
            1., KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, 1., 1., 1.
        ])
    
    def __call__(self, data):
        data.y = data.y / self.conversion
        return data