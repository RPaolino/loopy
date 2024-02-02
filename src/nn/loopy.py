
import math
import torch
from torch import Tensor
import torch.nn
from torch.nn import (
    Module, 
    ModuleList, 
    Linear, 
    Parameter,
    init
)
import torch.nn.functional as F
from torch_scatter import segment_csr
from torch_geometric.data import Data
from typing import Callable, Union


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LoopyLayer(Module):
    """
    Notes: 
        For reproducibility, we use ``segment_csr`` instead of ``scatter_sum``.
        The forward behaviour is the same, but the arguments and the backward 
        behaviour are different. Specifically, the arguments of ``segment_csr`` is
        ``indptr``, a compressed version of the indices, that is, the position 
        where a new index appear
        ind = [0, 0, 0, 1, 1, 2] → ptr = [0, 3, 5, 6]
        Therefore, ``ind`` must be sorted in ascending order.
        
        The backward of ``scatter_sum`` is non-deterministic when indices are not unique:
        one of the values from ``src`` will be picked arbitrarily and it will be 
        propagated to all locations in the source that correspond to the same index!
        Hence, the gradient will be incorrect. In contrast to ``scatter``,
        ``segment_csr`` is fully-deterministic.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_edge_features: int,
        r: int, 
        nonlinearity: str = "relu", 
        norm: str ="bn",
        shared: bool = False,
        use_edge_attr: bool = False
    ):
        super().__init__()
        self.r = r
        self.shared = shared
        self.use_edge_attr = use_edge_attr
        
        # Define mlps
        if shared:
            num_convs = 1
        else:
            num_convs = r
        self.eps = Parameter(torch.zeros(1))
        self.r_eps = Parameter(torch.zeros(r+1))
        self.convs = ModuleList([])
        self.num_embeddings = int(math.ceil((r+1)/2))+1
        # The first layer will just process x and atomic_type of the direct
        # neighbors, without applying a final nn
        for i in range(num_convs):
            self.convs.append(
                CustomGINConv(
                    in_channels=in_channels,
                    nn=MLP(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        num_layers=2
                    ),
                    num_embeddings=self.num_embeddings,
                    train_eps=True
                )
            )      
        if self.use_edge_attr:
            self.edge_encoder = MLP(
                in_channels=num_edge_features,
                out_channels=out_channels,
                num_layers=2,
                norm=norm,
                nonlinearity=nonlinearity
            )
        self.conv_final = MLP(
            in_channels=in_channels,
            out_channels=out_channels,
            num_layers=2,
            norm=norm,
            nonlinearity=nonlinearity
        )
    
    def forward(
        self,
        x: Tensor, 
        edge_weight: Union[Tensor, None],
        data: Data
    ) -> Tensor:
        if self.use_edge_attr:
            edge_weight = self.edge_encoder(edge_weight)
        x = x.float()
        r_contribution = 0
        for L in range(self.r+1):
            # if --lazy, we need to compute the permutations of symple cycles
            # to get the r-neighborhoods
            if "loopyA" in data:
                loopyNL = torch.cat(
                    [data[f"loopyN{L}"].roll(shifts=shift, dims=0) for shift in range(L+2)],
                    dim=1
                )
                sorted_idx = loopyNL[0].argsort() #segment_csr needs sorted indices
                loopyNL = loopyNL[:, sorted_idx]
            else:
                loopyNL = data[f"loopyN{L}"]
            if loopyNL.shape[1]>0:
                current_contribution = x[loopyNL[1:]] 
                if self.use_edge_attr:
                    if "loopyE" in data:
                        edge_attr_idx = torch.tensor(
                            [
                                [data[f"loopyE"][source, target] for source, target in zip(loopyNL[0].tolist(), row)] for row in loopyNL.tolist()
                            ],
                            dtype=int
                        ).to(device)
                    else:
                        edge_attr_idx = data[f"loopyE{L}"]
                    current_contribution = (
                        current_contribution 
                        + edge_weight[edge_attr_idx]
                    )
                if L==0:
                    current_contribution = current_contribution.squeeze(0)
                else:
                    if self.shared:
                        current_conv = self.convs[0]
                    else:
                        current_conv = self.convs[L-1]
                    if "loopyA" in data:
                        atomic_type = torch.tensor(
                            [
                                [data[f"loopyA"][source, target] for source, target in zip(loopyNL[0].tolist(), row)] for row in loopyNL.tolist()
                            ],
                            dtype=int
                        ).to(device)
                    else:
                        atomic_type = data[f"loopyA{L}"]
                    current_contribution = current_conv(
                        x=current_contribution,
                        atomic_type=atomic_type[1:]
                    )
                ptr = torch._convert_indices_from_coo_to_csr(
                    loopyNL[0], 
                    size=x.shape[0]
                )
                current_contribution = segment_csr(
                    src=current_contribution, 
                    indptr=ptr
                    # dim=0,
                    # dim_size=x.shape[0]
                )
                r_contribution = (
                    r_contribution + (1+self.r_eps[L]) * current_contribution
                )
        # Final mlp on the sum of all contributions
        x = self.conv_final(
            (1 + self.eps) * x 
            + r_contribution 
        )
        return x
    
class MLP(Module):
    def __init__(
        self, 
        in_channels, 
        out_channels,
        num_layers,
        nonlinearity="relu",
        norm="Identity",
        
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.bound = math.sqrt(1./in_channels)
        self.initial_weight = Parameter(
            torch.zeros((in_channels, out_channels))
        )
        self.final_weight = Parameter(
            torch.zeros((num_layers-1, out_channels, out_channels))
        )
        self.initial_bias = Parameter(
            torch.zeros((out_channels))
        )
        self.final_bias = Parameter(
            torch.zeros((num_layers-1, out_channels))
        )
        self.norm = getattr(torch.nn, norm)(out_channels)
        self.nonlinearity = get_nonlinearity(nonlinearity, False)
        self.reset_parameters()
        
    def reset_parameters(self):
        init.kaiming_uniform_(self.initial_weight, a=math.sqrt(5))
        init.uniform_(self.initial_bias, -self.bound, self.bound)
        init.kaiming_uniform_(self.final_weight, a=math.sqrt(5))
        init.uniform_(self.final_bias, -self.bound, self.bound)

    def forward(self, x):
        x = torch.matmul(x, self.initial_weight) + self.initial_bias
        for (A, b) in zip(self.final_weight, self.final_bias):
            x = self.norm(x)
            x = self.nonlinearity(x)
            x = torch.matmul(x, A) + b     
        return x

    def __repr__(self):
        return self.__class__.__name__

class CustomGINConv(Module):
    r"""
    A GINConv layer implementing ``custom_propagate``.
    """      
    def __init__(
        self, 
        nn: Callable, 
        in_channels: int,
        num_embeddings: int = 100,
        train_eps: bool = True
    ):
        super().__init__()
        self.nn = nn
        self.eps = Parameter(torch.ones(1), requires_grad=train_eps)
        self.embedding = torch.nn.Embedding(num_embeddings, in_channels)
        self.transform_before_conv = Linear(2*in_channels, in_channels)
        self.reset_parameters()
    
    def reset_parameters(self):
        self.nn.reset_parameters()
        self.embedding.reset_parameters()
        self.transform_before_conv.reset_parameters()

    def forward(self, x, atomic_type):
        out = self.transform_before_conv(
            torch.cat(
                [
                    x, 
                    self.embedding(atomic_type)
                ], 
            dim=-1)
        )
        out = custom_propagate(out)
        out = self.nn(
            (1+self.eps) * x + out
        )
        return out.sum(0)
    
def custom_propagate(x: Tensor) ->Tensor:
    r"""
    In a path each node is connected to the previous and the next. 
    Therefore, summing over neighborhood is equivalent to a convolution with an 
    appropriate kernel.
    
    Example:
        For simplicity, suppose the first channel is
        x = [[1, 2, 3, 4],
             [5, 6, 7, 8]]
        To be a valid input, it needs padding
        x = [[0, 0, 0, 0],
             [1, 2, 3, 4],
             [5, 6, 7, 8],
             [0, 0, 0, 0]]
        Finally, we apply a convolution with kernel
        k = [[1],
             [0],
             [1]]
        and stride 1 to get
        x = [[5, 6, 7, 8],
             [1, 2, 3, 4]]
        which is the expected behaviour.
    """
    kernel = torch.tensor([1., 0., 1.], device=device).reshape(1, 1, 1, 1, 3)
    # x has dimension (path_length, num. paths, num. features)
    # but conv3d expects num.features as second dimension, and length of 
    # batch as first.
    out = x.transpose(0, 2).unsqueeze(0)
    out = F.conv3d(
        out, 
        kernel, 
        padding="same"
    )
    # Parsing to dim. (path_length, num. paths, num. features)
    out = out.squeeze(0).transpose(0, 2)
    return out

def get_nonlinearity(
    nonlinearity: str, 
    return_module: bool = True
    ) -> Union[Module, Callable]:
    """
    Args:
        nonlinearity (str): name of nonlinearity.
        return_module (bool, optional): if True, it returns the Module from
        torch.nn; otherwise, it returns the function from torch.nn.functionals.
    """
    if nonlinearity == 'relu':
        module = torch.nn.ReLU()
        function = torch.nn.functional.relu
    elif nonlinearity == 'elu':
        module = torch.nn.ELU()
        function = torch.nn.functional.elu
    elif nonlinearity == 'gelu':
        module = torch.nn.GELU()
        function = torch.nn.functional.gelu
    elif nonlinearity == 'id':
        module = torch.nn.Identity()
        function = lambda x: x
    elif nonlinearity == 'sigmoid':
        module = torch.nn.Sigmoid()
        function = torch.nn.functional.sigmoid
    elif nonlinearity == 'tanh':
        module = torch.nn.Tanh()
        function = torch.tanh
    else:
        raise NotImplementedError(f'Nonlinearity {nonlinearity} not supported.')
    if return_module:
        return module
    else:
        return function