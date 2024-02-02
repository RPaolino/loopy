import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data

from ..r_neighborhood import rNeighborhood

def test_star():
    edgelist = [
        (0, 1), (0, 2), (0, 3), (0, 4),
        (1, 0), (2, 0), (3, 0), (4, 0)
    ]
    G = nx.from_edgelist(edgelist)

    data = Data(
        edge_index=torch.LongTensor(edgelist).t().contiguous(),
        num_nodes=5
    )
    
    transform  = rNeighborhood(r=2)
    data = transform(data)
    expected_maximal_hop = 1
    assert (data[f"loopyA0"]>expected_maximal_hop).sum()==0, (
        f"Incorrect hops: expected to be ≤{expected_maximal_hop }, "
        f" {np.max(data[f'loopyA0'])}."
    )
    assert (data[f"loopyA0"][0]==0).sum()==data[f"loopyA0"].shape[1], (
        f"Incorrect hops: first row expected to be all zeroes."
    )
    assert (data[f"loopyA0"][1]==1).sum()==data[f"loopyA0"].shape[1], (
        f"Incorrect hops: second row expected to be all ones."
    )
    assert (data[f"loopyA0"][-1]==1).sum()==data[f"loopyA0"].shape[1], (
        f"Incorrect hops: penultimate row expected to be all ones."
    )
    for ncol in range(data["loopyN0"].shape[1]):
        for nrow, (u, v) in enumerate(zip(data["loopyN0"][:-1, ncol], data["loopyN0"][1:, ncol])):
            idx = torch.logical_and(
                data.edge_index[0]==u.item(), data.edge_index[1]==v.item()
            ).nonzero(as_tuple=True)[0]
            assert idx==data[f"loopyE0"][nrow, ncol], (
                f"Incorrect loopyE0."
            )
    for center in range(data.num_nodes):
        idx = data["loopyN0"][0]==center
        paths = data["loopyN0"][:, idx]
        if center==0:
            assert paths.shape == (2, 4), (
                f"Shape of direct neighbors not correct:"
                f" expected (2, 4) for center node {center}, got {paths.shape}."
            )
        else:
            assert paths.shape == (2, 1), (
                f"Shape of direct neighbors not correct:"
                f" expected (2, 1) for center node {center}, got {paths.shape}."
            )
        
def test_triangle():
    edgelist = [
        (0, 1), (1, 2), (2, 0),
        (1, 0), (2, 1), (0, 2)
    ]
    G = nx.from_edgelist(edgelist)
    data = Data(
        edge_index=torch.LongTensor(edgelist).t().contiguous(),
        num_nodes=3
    )
    r=1
    transform  = rNeighborhood(r=r)
    data = transform(data)
    for center in range(data.num_nodes):
        for L in range(r+1):
            expected_maximal_hop = int(np.ceil((L+1)/2))
            assert (data[f"loopyA{L}"]>expected_maximal_hop).sum()==0, (
                f"Incorrect loopyA{L}: expected to be ≤{expected_maximal_hop }, "
                f" {np.max(data[f'loopyA{L}'])}."
            )
            assert (data[f"loopyA{L}"][0]==0).sum()==data[f"loopyA{L}"].shape[1], (
                f"Incorrect hops: first row expected to be all zeroes."
            )
            assert (data[f"loopyA{L}"][1]==1).sum()==data[f"loopyA{L}"].shape[1], (
                f"Incorrect hops: second row expected to be all ones."
            )
            assert (data[f"loopyA{L}"][-1]==1).sum()==data[f"loopyA{L}"].shape[1], (
                f"Incorrect hops: penultimate row expected to be all ones."
            )
            for ncol in range(data[f"loopyN{L}"].shape[1]):
                for nrow, (u, v) in enumerate(zip(data[f"loopyN{L}"][:-1, ncol], data[f"loopyN{L}"][1:, ncol])):
                    idx = torch.logical_and(
                        data.edge_index[0]==u.item(), data.edge_index[1]==v.item()
                    ).nonzero(as_tuple=True)[0]
                    assert idx==data[f"loopyE{L}"][nrow, ncol], (
                        f"Incorrect loopyE{L}."
                    )
            idx = data[f"loopyN{L}"][0]==center
            paths = data[f"loopyN{L}"][:, idx]
            if L==0:
                assert paths.shape == (2, 2), (
                    f"Shape of direct neighbors not correct:"
                    f" expected (2, 2), got {paths.shape}."
                )
            elif L==1:
                assert paths.shape == (3, 2), (
                    f"Shape of loopyN{L} not correct: "
                    f"expected (3, 2), got {paths.shape}."
                )
    
def test_cycle():
    edgelist = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (1, 0), (2, 1), (3, 2), (0, 3)
    ]
    G = nx.from_edgelist(edgelist)
    data = Data(
        edge_index=torch.LongTensor(edgelist).t().contiguous(),
        num_nodes=4
    )
    r=2
    transform  = rNeighborhood(r=r)
    data = transform(data)
    for center in range(data.num_nodes):
        for L in range(r+1):
            expected_maximal_hop = int(np.ceil((L+1)/2))
            assert (data[f"loopyA{L}"]>expected_maximal_hop).sum()==0, (
                    f"Incorrect data loopyA{L}: expected to be ≤{expected_maximal_hop }, "
                    f" {np.max(data[f'loopyA{L}'])}."
                )
            assert (data[f"loopyA{L}"][0]==0).sum()==data[f"loopyA{L}"].shape[1], (
                f"Incorrect hops: first row expected to be all zeroes."
            )
            assert (data[f"loopyA{L}"][1]==1).sum()==data[f"loopyA{L}"].shape[1], (
                f"Incorrect hops: second row expected to be all ones."
            )
            assert (data[f"loopyA{L}"][-1]==1).sum()==data[f"loopyA{L}"].shape[1], (
                f"Incorrect hops: penultimate row expected to be all ones."
            )
            for ncol in range(data[f"loopyN{L}"].shape[1]):
                for nrow, (u, v) in enumerate(zip(data[f"loopyN{L}"][:-1, ncol], data[f"loopyN{L}"][1:, ncol])):
                    idx = torch.logical_and(
                        data.edge_index[0]==u.item(), data.edge_index[1]==v.item()
                    ).nonzero(as_tuple=True)[0]
                    assert idx==data[f"loopyE{L}"][nrow, ncol], (
                        f"Incorrect loopyE{L}."
                    )
            idx = data[f"loopyN{L}"][0]==center
            paths = data[f"loopyN{L}"][:, idx]
            if L==0:
                assert tuple(paths.shape) == (2, 2), (
                    f"Shape of direct neighbors not correct:"
                    f" expected (2, 2), got {tuple(paths.shape)}."
                )
            elif L==1:
                assert paths.shape[1]==0, f"There should be no path of length 1."
            elif L==2:

                assert tuple(paths.shape) == (4, 2), (
                    f"Shape of loopyN{L} of length 2 not correct: "
                    f"expected (4, 2), got {tuple(paths.shape)}."
                )
    expectedN2 = [
        [0, 1, 2, 3],
        [0, 3, 2, 1],
        [1, 2, 3, 0],
        [1, 0, 3, 2],
        [2, 3, 0, 1],
        [2, 1, 0, 3],
        [3, 0, 1, 2],
        [3, 2, 1, 0]
    ]
    print(data.loopyN2.t().tolist())
    assert sorted(data.loopyN2.t().tolist())==sorted(expectedN2), f"loopyN2 is wrong!"