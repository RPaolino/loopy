import numpy as np
import torch
from torch_geometric.data import Data

def custom_collate(data_list: list) -> Data:
    r"""It collates attribs.
    This function is needed because some attributes are dictionaries, hence, the 
    default collate wouldn't be able to collate them.
    
    Args:
        data_list (list of Data): list of data to collate.
    """
    # attribs is an iterable, with element ((key1, value1), ..., (key1, valueN))
    # where N is the length of data_list
    attribs = list(zip(*data_list))
    data_dict = {
        k: [] for k in data_list[0].keys()
    }
    num_nodes_offset = np.cumsum(
        [0]+[data.num_nodes for data in data_list]
    )
    num_edges_offset = np.cumsum(
        [0]+[data.edge_index.shape[1] for data in data_list]
    )
    for attrib in attribs:
        k = attrib[0][0]
        if k in ["edge_index"]:
            data_dict[k] =torch.cat(
                [
                    value+num_nodes_offset[ndata] for ndata, (_, value) in enumerate(attrib)
                ], 
                dim=1
            ) 
        elif k.startswith("loopyN"):
            data_dict[k] =torch.cat(
                [
                    value+num_nodes_offset[ndata] for ndata, (_, value) in enumerate(attrib)
                ], 
                dim=1
            )
        elif k.startswith("loopyE") and k!="loopyE":
            data_dict[k] = torch.cat(
                [
                    value+num_edges_offset[ndata] for ndata, (_, value) in enumerate(attrib)
                ], 
                dim=1
            )
        elif k=="loopyE":
            data_dict[k] = {
                (k[0]+num_nodes_offset[ndata], k[1]+num_nodes_offset[ndata]): v+num_edges_offset[ndata] for ndata, (_, value) in enumerate(attrib) for k, v in value.items()
            }
        elif k.startswith("loopyA") and k!="loopyA":
            data_dict[k] = torch.cat(
                [
                    value for _, value in attrib
                ], 
                dim=1
            )
        elif k=="loopyA":
            # distinction between pre_computing and lazy, in the lazy approach
            # the atomic type is a dictionary
            data_dict[k] = {
                (k[0]+num_nodes_offset[ndata], k[1]+num_nodes_offset[ndata]): v for ndata, (_, value) in enumerate(attrib) for k, v in value.items()
            }
        elif k in ["num_nodes", 
                   "num_graphs"]:
            continue
        else:
            data_dict[k] = torch.cat(
                [
                    value  for ndata, (_, value) in enumerate(attrib)
                ], 
                dim=0
            )
            if k=="x":
                # Collect which graph each row of x belongs to, for each graph in data_list
                data_dict["batch"] = torch.cat(
                    [
                        torch.tensor(
                            [ndata] * value.shape[0], 
                            dtype=torch.long
                        ) for ndata, (_, value) in enumerate(attrib)
                    ], 
                    dim=0
                )
    data_dict["num_graphs"] = len(data_list)
    data_dict["num_nodes"] = data_dict["x"].shape[0]
    # Create final output
    data = Data.from_dict(data_dict).to(data_list[0].edge_index.device)
    return data