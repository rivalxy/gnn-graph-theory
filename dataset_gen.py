from pynauty import Graph, autgrp
from sklearn.model_selection import train_test_split
import random
import torch
from torch_geometric.data import Data
from sympy.combinatorics import Permutation, PermutationGroup

from dataset.utils import is_paut, is_extensible, read_graphs_from_g6


MAX_EXAMPLES_NUM = 10
MAX_ATTEMPTS = 100


def gen_positive_examples(group: PermutationGroup, num_of_nodes: int, examples_num: int) -> list:
    seen_positives = set()
    positives = []
    attempts = 0

    while len(positives) < examples_num and attempts < MAX_ATTEMPTS * examples_num:
        attempts += 1
        perm = group.random().array_form
        nodes = list(range(num_of_nodes))
        if perm == nodes:
            continue  # skip identity

        p_aut_size = random.randint(
            max(3, num_of_nodes // 3), max(4, 2 * num_of_nodes // 3))
        domain = random.sample(nodes, p_aut_size)
        mapping = {i: perm[i] for i in domain}

        key = frozenset(mapping.items())
        if key in seen_positives:
            continue

        seen_positives.add(key)
        positives.append(mapping)
    return positives


#FIXME bad implementation
def gen_negative_examples(positives, num_of_nodes: int, ) -> list:
    negatives = []
    seen_negatives = set()

    for mapping in positives:
        u = random.choice(list(mapping.keys()))
        v_old = mapping[u]
        v_new = random.choice(
            [v for v in range(num_of_nodes) if v != v_old])
        neg_map = mapping.copy()
        neg_map[u] = v_new
        key = frozenset(neg_map.items())
        if key in seen_negatives:
            continue
        seen_negatives.add(key)
        negatives.append(neg_map)
    return negatives


def make_pyg_data(edge_list: list[tuple], num_of_nodes: int,  mapping: dict[int, int], label: int) -> Data:
    # 3 features: node_id, target_id, source_id
    x = torch.full((num_of_nodes, 3), -1.0, dtype=torch.float)

    # Give EVERY node its own identity
    for node in range(num_of_nodes):
        x[node, 0] = float(node) / num_of_nodes

    # Mark mapped nodes with bidirectional info
    for source, target in mapping.items():
        x[source, 1] = float(target) / num_of_nodes  # target_id
        x[target, 2] = float(source) / num_of_nodes  # source_id

    if len(edge_list) > 0:
        edges = []
        for u, v in edge_list:
            edges.append([u, v])
            edges.append([v, u])
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    y = torch.tensor([label], dtype=torch.float)
    data = Data(x=x, edge_index=edge_index, y=y)
    return data


def generate_paut_dataset(graphs: list[Graph]) -> list:
    """
    Generates partial automorphism mappings with their labels from a list of pynauty graphs.

    :param graphs: List of pynauty graphs.
    :returns: List of PyG Data objects containing partial automorphism mappings and labels.
    """

    positive_pyg_data = []
    negative_pyg_data = []

    for Graph, num_of_nodes, edge_list in graphs:
        generators_raw, grpsize1, grpsize2, _, _ = autgrp(Graph)
        group_size = grpsize1 * 10**grpsize2

        # ensure up to 10 examples per graph with 1:1 ratio of positive to negative examples
        examples_num = int(min(MAX_EXAMPLES_NUM, group_size))
        generators = [Permutation(g) for g in generators_raw]
        group = PermutationGroup(generators)

        positives = gen_positive_examples(
            group, num_of_nodes, examples_num)
        for mapping in positives:
            assert is_paut(edge_list, mapping)
            assert is_extensible(group, mapping)
            positive_pyg_data.append(make_pyg_data(
                edge_list, num_of_nodes, mapping, label=1))

        negatives = gen_negative_examples(positives, num_of_nodes)
        for mapping in negatives:
            assert is_paut(edge_list, mapping)
            assert not is_extensible(group, mapping)
            negative_pyg_data.append(make_pyg_data(
                edge_list, num_of_nodes, mapping, label=0))
            
    dataset = positive_pyg_data + negative_pyg_data
    return dataset


if __name__ == "__main__":
    positive_graphs = read_graphs_from_g6("dataset/positive_graphs.g6")
    graphs_train, graphs_val = train_test_split(positive_graphs, test_size=0.2)
    train_dataset = generate_paut_dataset(graphs_train)
    val_dataset = generate_paut_dataset(graphs_val)
    print(
        f"Generated {len(train_dataset)} train examples and {len(val_dataset)} val examples.")
