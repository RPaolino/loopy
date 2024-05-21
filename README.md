# Weisfeiler and Leman Go Loopy
+ [Code Structure](#code-structure)
    - [r-neighborhoods](#r-neighborhoods)
    - [Loopy Layer](#loopy_layer)
+ [Experiments](#experiments)

# Code Structure
You can use r-neighborhoods in your project by importing the following modules
```bash
├── src/
    └── data/
        └── custom_collate.py   # collate function that accounts for r-neighborhoods
    └── transforms/
        └── r_neighborhood.py   # build the r-neighborhood of each node in the graph
    └── nn/
        └── loopy.py   # class definition of loopy layers that process r-neighborhoods
```
and modifying them according to your needs.

## r-neighborhoods
Given an input graph $G$ and a node $v$, the ``r-neighborhood`` $\mathcal{N}_r(v)$ of $v$ is defined as the collection of simple paths of length $r$ between its distinct neighborhoods.
<center>
<img src="imgs/Nr.svg">
</center>

The computation of r-neighborhoods uses the ``networkx.simple_cycles`` function, which return simple cycles of the input graph $G$. The paths are then obtained as cyclic permutations of the simple cycles: the first node is the neighborhood center, while the others form the path.
<center>
<img src="imgs/Nr_computation.svg">
</center>

The paths are usually computed in the preprocessing step. However, this could lead to memory overload, especially when $G$ is dense. In order to prevent OOM, we also provide a ``--lazy`` flag, which postpone the computation of cyclic permutation to the forward step. In this way, we don't store all paths for each graph in the dataset; rather, we compute them on the flight.

---

## Loopy Layers
$r$-neighborhoods are then fed into a path-wise layer, which compute for each path an embedding. The embeddings are then processed together to get an embedding of the central node $v$.
<center>
<img src="imgs/lMPNN.svg">
</center>
In our code, we use GIN layers to process paths, as it is simple but maximally expressive on paths. You can choose a different neural architecture. Note that messages on paths are transmitted via ``torch.nn.functional.conv3d`` with kernel $[1, 0, 1]$, since only consecutive nodes are linked.

To limit the number of learnable parameters, we provide a ``--shared`` flag: it guarantees that in each loopy layer we have shared weights among paths of different lengths.

To implement the pooling operations, we use ``segment_csr``  instead of ``scatter``: the former is fully-deterministic, as noted in the [documentation](https://pytorch-scatter.readthedocs.io/en/latest/functions/segment_csr.html). When indices are not unique, the behavior of ``scatter`` is non-deterministic: one of the values from ``src`` will be picked arbitrarily and the gradient propagated to all elements with same index, resulting in an incorrent gradient computation.

---

# Experiments

The hyperparams used for the experiments can be retrieved from the folder
```bash
├── scripts/
```
You can reproduce the results by typing
```bash
bash scripts/<dataset_name>.sh
```
or you can specify your own configuration as
```bash
python run_model.py --dataset zinc_subset --r 5
```
For ``subgraphcount`` [[2]](#2), you need to specify the target motiv $n$
<center>
<table>
  <tr>
    <td><center>n</center></td>
    <td><center>0</center></td>
    <td><center>1</center></td>
    <td><center>2</center></td>
    <td><center>3</center></td>
    <td><center>4</center></td>
    <td><center>5</center></td>
    <td><center>6</center></td>
    <td><center>7</center></td>
    <td><center>8</center></td>
  </tr>
  <tr>
    <td><center>F</center></td>
    <td><center><img src="imgs/chordal_4.svg"></center></td>
    <td><center><img src="imgs/boat.svg"></center></td>
    <td><center><img src="imgs/chordal_6.svg"></center></td>
    <td><center><img src="imgs/cycle_3.svg"></center></td>
    <td><center><img src="imgs/cycle_4.svg"></center></td>
    <td><center><img src="imgs/cycle_5.svg"></center></td>
    <td><center><img src="imgs/cycle_6.svg"></center></td>
    <td><center><img src="imgs/chordal_4.svg"></center></td>
    <td><center><img src="imgs/chordal_5.svg"></center></td>
  </tr>
<tr>
    <td><center> </center></td>
    <td colspan="3"><center>hom(F, G)</center></td>
    <td colspan="6"><center>sub(F, G)</center></td>
  </tr>
</table>
</center>
by typing

```bash
python run_model.py --dataset subgraphcount_2 --r 1
```
The first three motifs are used to test against homomorphism-counts, the latter six against subgraph-counts. The preprocessing of the dataset is done following [the official repo](https://github.com/subgraph23/homomorphism-expressivity) of [[3]](#3).

Similarly, you can specify the regression target of ``qm9`` by ``qm9_<n>`` where $n$ is the columns index of the target. 

For ``brec`` [[1]](#1), you need to specify the name of the raw file, i.e., ``brec_<name>`` where name is one among ``basic``, ``extension``, ``regular``, ``4vtx`` (4-vertex condition), ``dr`` (distance regular), ``str`` (strongly regular), and ``cfi``. 

Moreover, ``exp_iso`` is the name given to ``exp`` when the task is to count the number of indistinguishable pairs.

# References
<a id="1">[1]</a> Yanbo Wang et al. "Towards better evaluation of gnn expressiveness with brec dataset." arXiv preprint arXiv:2304.07702 (2023).

<a id="2">[2]</a> Lingxiao Zhao et al. "From stars to subgraphs: Uplifting any gnn with local structure awareness." In International Conference on Learning Representations, 2022.

<a id="3">[3]</a> Bohang Zhang et al. "Beyond Weisfeiler-Lehman: A Quantitative Framework for GNN Expressiveness." In International Conference on Learning Representations, 2024.
