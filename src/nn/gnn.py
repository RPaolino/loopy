from functools import partial
import torch
from torch.nn import (
    Module, 
    ModuleList,
    Linear,
    Embedding
)
from torch.nn.init import xavier_uniform_
from torch_scatter import segment_csr


from .loopy import (
    LoopyLayer, 
    get_nonlinearity
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LearnableLookUpTable(Module):
    r"""
    # Since the node features on ``zinc`` are integers, we use an Embedding layer,
    i.e., a learnable look-up table which has the benefit that the embedding is 
    not necessarly proportional to the key as in a Linear layer.
    """
    def __init__(
        self, 
        hidden_channels: int, 
        feature_dims = None
    ):
        super().__init__()
        self.embedding = ModuleList([])
        for dim in feature_dims:
            emb = Embedding(
                num_embeddings=dim,
                embedding_dim=hidden_channels
            )
            xavier_uniform_(emb.weight.data)
            self.embedding.append(emb)

    def forward(self, x):
        x_embedding = 0
        for i in range(x.shape[1]):
            x_embedding += self.embedding[i](x[:,i].long())
        return x_embedding
    
class GNN(Module):
    r"""
    Creating the loopy neural network.
    If ``num_node_encoder_layers`` (respectively ``num_edge_encoder_layers``) 
    is greater than 0, then node (respectively, edge) features are encoded with:
    - a sequence of Linear layers, if the features are not categorical, i.e.,
    if the dtype is not torch.int64;
    - a first Embedding layer, followed by Linear layers if the features' dtype 
    is torch.int64.
    A ``num_layers`` layers of loopy message passing layers are then applied, 
    followed by graph pooling and finally ``num_decoder_layers`` Linear layers 
    to compute the prediction.
    
    Notes:
        To use edge atributes, you need to set ``use_edge_attr`` to True.
    """
    
    def __init__(
        self, 
        dataset,
        hidden_channels: int = 64,
        out_channels: int = 10,
        num_node_encoder_layers: int = 0, 
        num_edge_encoder_layers: int = 0, 
        num_layers: int = 3, 
        num_decoder_layers: int  = 1, 
        norm: str = "Identity",
        dropout: float = 0., 
        nonlinearity: str = "relu",
        graph_pooling: str = "sum",   
        use_edge_attr: bool = False,     
        r: int = 1,
        shared: bool = False,
    ):
        super().__init__()
        # Node encoder
        self.node_encoder = ModuleList([])
        self.num_node_encoder_layers = num_node_encoder_layers
        for i in range(num_node_encoder_layers):
            if i==0:
                if dataset.x.dtype == torch.int64:
                    self.node_encoder.append(
                        LearnableLookUpTable(
                            feature_dims=(
                                dataset._data.x.max(0).values.to(int)
                                +1
                            ).tolist(),
                            hidden_channels=hidden_channels
                        )
                    )
                else:
                    self.node_encoder.append(
                        Linear(dataset.num_node_features, hidden_channels)
                    )
            else:
                self.node_encoder.append(
                    Linear(hidden_channels, hidden_channels)
                )
            
        # Edge encoder
        self.edge_encoder = ModuleList([])
        self.num_edge_encoder_layers = num_edge_encoder_layers
        self.use_edge_attr = use_edge_attr
        if use_edge_attr:
            for i in range(num_edge_encoder_layers):
                if i==0:
                    if dataset.x.dtype == torch.int64:
                        self.edge_encoder.append(
                            LearnableLookUpTable(
                                feature_dims=(
                                    dataset._data.edge_attr.max(0).values.to(int)
                                    +1
                                ).tolist(),
                                hidden_channels=hidden_channels
                            )
                        )
                    else:
                        self.edge_encoder.append(
                            Linear(dataset.num_edge_features, hidden_channels)
                        )
                else:
                    self.edge_encoder.append(
                        Linear(hidden_channels, hidden_channels)
                    )
        # Layers
        self.num_layers = num_layers
        self.convs = ModuleList([])
        self.norms = ModuleList([])
        assert num_layers>=1, f'Num. layer should be ≥ 1, got {num_layers}'
        for i in range(num_layers):
            if i==0:
                in_size = dataset.num_node_features if num_node_encoder_layers==0 else hidden_channels
            else:
                in_size = hidden_channels
            if i==num_layers-1:
                out_size = out_channels  if num_decoder_layers==0  else hidden_channels
            else:
                out_size = hidden_channels
            self.convs.append(
                LoopyLayer(
                    in_channels=in_size,
                    out_channels=out_size,
                    num_edge_features=dataset.num_edge_features if (num_edge_encoder_layers==0) else hidden_channels,
                    r=r,
                    nonlinearity=nonlinearity,
                    norm=norm,
                    shared=shared,
                    use_edge_attr=use_edge_attr
                )
            )            
            # Normalization
            self.norms.append(
                getattr(torch.nn, norm)(out_size)
            )  
        # Nonlinearity
        self.nonlinearity = get_nonlinearity(nonlinearity)
        # Dropout      
        self.dropout = torch.nn.Dropout(p=dropout)
        # Pooling function to generate whole-graph embeddings
        self.graph_pooling = partial(
            reproducible_pooling,
            reduce=graph_pooling
        )
        # Decoding mlps
        self.decoder = ModuleList([])
        self.num_decoder_layers = num_decoder_layers
        for i in range(num_decoder_layers):
            in_size = hidden_channels
            if i==num_decoder_layers-1:
                out_size = out_channels
            else:
                out_size = hidden_channels
            self.decoder.append(
                Linear(in_size, out_size)
            )

    def forward(self, batched_data):
        batched_data = batched_data.to(device)
        node_representation = batched_data.x.float()
        # Encoding node features
        for n_layer, layer in enumerate(self.node_encoder):
            node_representation = layer(node_representation)
            if n_layer != self.num_node_encoder_layers-1:
                node_representation = self.nonlinearity(node_representation)
        # Encoding edge_attr
        edge_representation = batched_data.edge_attr.float()
        for n_layer, layer in enumerate(self.edge_encoder):
            edge_representation = layer(edge_representation)
            if n_layer != self.num_edge_encoder_layers-1:
                edge_representation = self.nonlinearity(edge_representation)
        # Convolutional layers
        for layer in range(self.num_layers):
            node_representation = self.convs[layer](
                x=node_representation,
                edge_weight=edge_representation,
                data=batched_data
            )
            # Normalization layer
            node_representation = self.norms[layer](node_representation)
            # Nonlinearity
            if (layer != self.num_layers-1):
                node_representation = self.nonlinearity(node_representation)
            # Dropout
            node_representation = self.dropout(node_representation)
        # Getting a representation of the whole graph
        graph_representation = self.graph_pooling(
            node_representation, 
            idx=batched_data.batch
        )
        # Decoding
        for n_layer, layer in enumerate(self.decoder):
            graph_representation = layer(graph_representation)
            if n_layer != self.num_decoder_layers-1:
                graph_representation = self.nonlinearity(graph_representation)
        return graph_representation
    

def reproducible_pooling(x, idx, reduce):
    """
    Function implementing segment_csr.
    
    Args:
        x (Tensor): tensor whose value needs to be reduced.
        idx (Tensor): tensor of indices. It must besorted.
        reduce (str): which reduction method to apply, such as ``min``, ``max``, 
            ``mean`` or ``sum``.
    """
    ptr = torch._convert_indices_from_coo_to_csr(
        idx, 
        size=int(idx.max())+1
    )
    out = segment_csr(
        src=x, 
        indptr=ptr,
        reduce=reduce
    )
    return out