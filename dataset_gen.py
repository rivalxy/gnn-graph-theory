import random
import torch

from pynauty import Graph, autgrp
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from sympy.combinatorics import Permutation, PermutationGroup

from dataset.utils import is_paut, is_extensible, read_graphs_from_g6


MAX_EXAMPLES_NUM = 10
MAX_ATTEMPTS = 100


def gen_positive_examples(group: PermutationGroup, num_of_nodes: int, examples_num: int) -> list:
    seen_positives = set()
    positives = []
    attempts = 0
    nodes = list(range(num_of_nodes))

    while len(positives) < examples_num and attempts < MAX_ATTEMPTS * examples_num:
        attempts += 1
        perm = group.random().array_form
        if all(i == p for i, p in enumerate(perm)):
            continue  # skip trivial case

        p_aut_size = random.randint(
            min(3, num_of_nodes // 3), 4 * num_of_nodes // 5)
        domain = random.sample(nodes, p_aut_size)
        mapping = {i: perm[i] for i in domain}

        key = frozenset(mapping.items())
        if key in seen_positives:
            continue

        seen_positives.add(key)
        positives.append(mapping)
    return positives


def negatives_blocking(group: PermutationGroup,
                       examples_num: int,
                       num_of_nodes: int,
                       edge_list: list[tuple]
                       ) -> list:
    negatives = []
    seen_negatives = set()
    attempts = 0
    nodes = list(range(num_of_nodes))

    adjacency_list = {}
    for u, v in edge_list:
        adjacency_list.setdefault(u, set()).add(v)
        adjacency_list.setdefault(v, set()).add(u)

    while len(negatives) < examples_num and attempts < MAX_ATTEMPTS * MAX_EXAMPLES_NUM:
        attempts += 1
        perm = group.random().array_form
        if all(i == p for i, p in enumerate(perm)):
            continue  # skip trivial case

        p_aut_size = random.randint(
            min(3, num_of_nodes // 3), 4 * num_of_nodes // 5)
        domain = random.sample(nodes, p_aut_size)
        mapping = {i: perm[i] for i in domain}

        blocked_mapping = block_automorphism(mapping, num_of_nodes, adjacency_list)

        if blocked_mapping is None:
            continue

        if not is_paut(edge_list, blocked_mapping):
            continue
        if is_extensible(group, blocked_mapping):
            continue

        # ensure uniqueness
        key = frozenset(blocked_mapping.items())
        if key in seen_negatives:
            continue

        seen_negatives.add(key)
        negatives.append(blocked_mapping)

    return negatives


def block_automorphism(positive: dict, num_of_nodes: int, adj: dict) -> dict:
    nodes = list(range(num_of_nodes))

    unmapped_nodes = [n for n in nodes if n not in positive]
    targets = [n for n in nodes if n not in positive.values()]

    if not unmapped_nodes or not targets:
        return None

    random.shuffle(unmapped_nodes)
    random.shuffle(targets)

    for node in unmapped_nodes[:min(5, len(unmapped_nodes))]:
        node_neighbors = adj.get(node, set())

        for target in targets[:min(5, len(targets))]:
            target_neighbors = adj.get(target, set())

            test_map = positive.copy()
            test_map[node] = target

            valid = True
            for neighbor in node_neighbors:
                if neighbor in test_map:
                    if test_map[neighbor] not in target_neighbors:
                        valid = False
                        break

            if valid:
                return test_map
            
    return None


# TODO
def gen_negative_examples(group: PermutationGroup,
                          examples_num: int,
                          num_of_nodes: int,
                          edge_list: list[tuple]
                          ) -> list:

    negatives = negatives_blocking(
        group, examples_num, num_of_nodes, edge_list)

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


def generate_paut_dataset(pynauty_graphs: list[Graph]) -> list:
    """
    Generates partial automorphism mappings with their labels from a list of pynauty graphs.

    :param graphs: List of pynauty graphs.
    :returns: List of PyG Data objects containing partial automorphism mappings and labels.
    """

    positive_pyg_data = []
    negative_pyg_data = []

    for pynauty_graph, num_of_nodes, edge_list in pynauty_graphs:
        generators_raw, grpsize1, grpsize2, _, _ = autgrp(pynauty_graph)
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

        negatives = gen_negative_examples(
            group, examples_num, num_of_nodes, edge_list)
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

    torch.save(train_dataset, "dataset/train_dataset.pt")
    torch.save(val_dataset, "dataset/val_dataset.pt")

    print(
        f"Generated {len(train_dataset)} train examples and {len(val_dataset)} val examples.")
