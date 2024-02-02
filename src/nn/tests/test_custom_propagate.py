import torch

from ...loopy import custom_propagate

def test_custom_propagate():
    """
    Given node features x of dim. (path length + 1, num. paths, num. features),
    the function ``custom_propagate`` is supposed to add features of neighbouring
    nodes, i.e., for each center node c and each feature dimension f it performs
    x[c-1, :, f] + x[c+1, :, f] taking care of extreme points.
    """
    L = 4
    num_paths = 30
    num_features = 64
    x = torch.randn((L, num_paths, num_features))
    expected = torch.zeros_like(x)
    for f in range(num_features):
        for center in range(L):
            if center==0:
                expected[center, :, f] = x[center+1, :, f]
            elif center==L-1:
                expected[center, :, f] = x[center-1, :, f]
            else:
                expected[center, :, f] = (
                    x[center-1, :, f] +  x[center+1, :, f]
                )
    out = custom_propagate(x)
    assert (expected-out).abs().max().item()<1e-10, (
        f"custom_propagate not behaving as expected!"
    )