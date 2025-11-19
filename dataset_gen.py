import networkx as nx
from pynauty import Graph, autgrp
from sklearn.model_selection import train_test_split
import random
import torch
from torch_geometric.data import Data
from sympy.combinatorics import Permutation, PermutationGroup


MAX_EXAMPLES_NUM = 30
MAX_ATTEMPTS = 100


def read_graphs_from_g6(file_path: str) -> list[Graph]:
    """
    Reads graphs from a .g6 file and converts them to igraph format.

    :param file_path: Path to the .g6 file.
    :returns: List of igraph graphs.
    """

    graphs = nx.read_graph6(file_path)
    pynauty_graphs = []
    for g in graphs:
        n = int(g.number_of_nodes())
        new_g = Graph(n)
        new_g.set_adjacency_dict(dict(g.adjacency()))
        pynauty_graphs.append((new_g, n, g.edges()))
    return pynauty_graphs


def generate_partial_automorphism_graphs(graphs: list[Graph]) -> list:
    """
    Generates partial automorphism graphs from a list of igraph graphs.

    :param graphs: List of igraph graphs.
    :returns: TODO
    """

    dataset = []

    for G, n, edge_list in graphs:
        gens_raw, group_size, _, _, _ = autgrp(G)

        # ensure 10-30 examples per graph with 1:1 ratio of positive to negative examples
        examples_num = int(min(MAX_EXAMPLES_NUM, group_size))
        gens = [Permutation(g) for g in gens_raw]
        group = PermutationGroup(gens)

        seen = set()

        # positive examples
        positives = []
        for _ in range(examples_num):
            perm = group.random().array_form
            k = random.randint(3, min(6, n))
            domain = random.sample(range(n), k)
            mapping = {i: perm[i] for i in domain}
            key = frozenset(mapping.items())
            if key in seen:
                continue
            seen.add(key)
            positives.append(mapping)
            dataset.append(_make_data(edge_list, n, mapping, 1))

        # negative examples
        for mapping in positives:
            u = random.choice(list(mapping.keys()))
            v_old = mapping[u]
            v_new = random.choice([v for v in range(n) if v != v_old])
            neg_map = mapping.copy()
            neg_map[u] = v_new
            key = frozenset(neg_map.items())
            if key in seen:
                continue
            seen.add(key)
            dataset.append(_make_data(edge_list, n, neg_map, label=0))

    return dataset


def _make_data(edge_list: list[tuple], n: int,  mapping: dict[int, int], label: int) -> Data:
    x = torch.zeros((n, 2), dtype=torch.float)
    map_tensor = torch.full((n,), -1, dtype=torch.long)

    for i, j in mapping.items():
        x[i, 0] = 1.0
        x[j, 1] = 1.0
        map_tensor[i] = j

    if len(edge_list) > 0:
        edges = []
        for u, v in edge_list:
            edges.append([u, v])
            edges.append([v, u])
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    y = torch.tensor([label], dtype=torch.float)
    data = Data(x=x, edge_index=edge_index, y=y, mapping=map_tensor)

    return data


if __name__ == "__main__":
    all_graphs = read_graphs_from_g6("dataset/2000_raw_graphs.g6")
    print(f"Average number of nodes: {sum(n for _, n, _ in all_graphs) / len(all_graphs)}")
    graphs_train, graphs_val = train_test_split(all_graphs, test_size=0.2)
    train_dataset = generate_partial_automorphism_graphs(graphs_train)
    val_dataset = generate_partial_automorphism_graphs(graphs_val)
    print(
        f"Generated {len(train_dataset)} train examples and {len(val_dataset)} val examples.")
