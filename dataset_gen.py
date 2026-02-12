import random
import torch

from pynauty import Graph, autgrp
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from sympy.combinatorics import Permutation, PermutationGroup
from collections import defaultdict

from utils import is_paut, is_extensible, read_graphs_from_g6, paut_sizes_to_csv

MAX_EXAMPLES_NUM = 10
MAX_ATTEMPTS = 100

paut_sizes = defaultdict(list)


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
        positives.append((mapping, p_aut_size))
    return positives


def negatives_blocking(group: PermutationGroup,
                       examples_num: int,
                       num_of_nodes: int,
                       adjacency_list: dict[int, set]
                       ) -> list:
    negatives = []
    seen_negatives = set()
    attempts = 0
    nodes = list(range(num_of_nodes))
    maximum_size = 4 * num_of_nodes // 5

    while len(negatives) < examples_num and attempts < MAX_ATTEMPTS * MAX_EXAMPLES_NUM:
        attempts += 1
        perm = group.random().array_form
        if all(i == p for i, p in enumerate(perm)):
            continue  # skip trivial case

        p_aut_size = random.randint(
            min(3, num_of_nodes // 3), maximum_size)
        p_aut_size -= 1
        original_paut_size = p_aut_size
        domain = random.sample(nodes, p_aut_size)
        mapping = {i: perm[i] for i in domain}

        blocked_mapping = block_automorphism(
            mapping, num_of_nodes, adjacency_list)

        if blocked_mapping is None:
            continue

        p_aut_size += 1

        if random.random() < 0.8 and p_aut_size < maximum_size:
            new_mapping = block_automorphism(blocked_mapping, num_of_nodes, adjacency_list)

            if new_mapping is None:
                continue

            blocked_mapping = new_mapping 
            p_aut_size += 1

        while random.random() < 0.5 and p_aut_size < maximum_size:
            new_mapping = block_automorphism(
                blocked_mapping, num_of_nodes, adjacency_list)

            if new_mapping is None:
                break

            blocked_mapping = new_mapping
            p_aut_size += 1

        if not is_paut(adjacency_list, blocked_mapping):
            continue
        if is_extensible(group, blocked_mapping):
            continue

        # ensure uniqueness
        key = frozenset(blocked_mapping.items())
        if key in seen_negatives:
            continue

        seen_negatives.add(key)
        extension_size = p_aut_size - original_paut_size
        negatives.append(
            (blocked_mapping, original_paut_size, extension_size))

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
                          adjacency_list: dict[int, set]
                          ) -> list:

    negatives = negatives_blocking(
        group, examples_num, num_of_nodes, adjacency_list)

    return negatives


def make_pyg_data(tensor_edge_index: torch.Tensor, num_of_nodes: int,  mapping: dict[int, int], label: int) -> Data:
    # 3 features: node_id, target_id, source_id
    x = torch.full((num_of_nodes, 3), -1.0, dtype=torch.float)

    # Give EVERY node its own identity
    for node in range(num_of_nodes):
        x[node, 0] = float(node) / num_of_nodes

    # Mark mapped nodes with bidirectional info
    for source, target in mapping.items():
        x[source, 1] = float(target) / num_of_nodes  # target_id
        x[target, 2] = float(source) / num_of_nodes  # source_id

    y = torch.tensor([label], dtype=torch.float)
    data = Data(x=x, edge_index=tensor_edge_index, y=y)
    return data


def build_edge_index(adjacency_list: dict[int, set]) -> torch.Tensor:
    if len(adjacency_list) == 0:
        return torch.empty((2, 0), dtype=torch.long)

    edges = []
    for u, neighbors in adjacency_list.items():
        for v in neighbors:
            edges.append([u, v])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index


def generate_paut_dataset(pynauty_graphs: list[Graph], dataset_type: str) -> list:
    """
    Generates partial automorphism mappings with their labels from a list of pynauty graphs.

    :param graphs: List of pynauty graphs.
    :param dataset_type: Type of the dataset (e.g., "train" or "val").
    :returns: List of PyG Data objects containing partial automorphism mappings and labels.
    """

    positive_pyg_data = []
    negative_pyg_data = []

    for pynauty_graph, num_of_nodes, adjacency_list in pynauty_graphs:
        generators_raw, grpsize1, grpsize2, _, _ = autgrp(pynauty_graph)
        group_size = grpsize1 * 10**grpsize2

        # ensure up to 10 examples per graph with 1:1 ratio of positive to negative examples
        examples_num = int(min(MAX_EXAMPLES_NUM, group_size))
        generators = [Permutation(g) for g in generators_raw]
        group = PermutationGroup(generators)

        tensor_edge_index = build_edge_index(adjacency_list)

        positives = gen_positive_examples(
            group, num_of_nodes, examples_num)

        for mapping, p_aut_size in positives:
            assert is_paut(adjacency_list, mapping)
            assert is_extensible(group, mapping)

            paut_sizes[num_of_nodes].append((p_aut_size, 0, dataset_type))
            positive_pyg_data.append(make_pyg_data(
                tensor_edge_index, num_of_nodes, mapping, label=1))

        negatives = gen_negative_examples(
            group, examples_num, num_of_nodes, adjacency_list)

        for mapping, original_paut_size, extension_size in negatives:
            assert is_paut(adjacency_list, mapping)
            assert not is_extensible(group, mapping)

            paut_sizes[num_of_nodes].append(
                (original_paut_size, extension_size, dataset_type))
            negative_pyg_data.append(make_pyg_data(
                tensor_edge_index, num_of_nodes, mapping, label=0))

    dataset = positive_pyg_data + negative_pyg_data
    return dataset


if __name__ == "__main__":
    positive_graphs = read_graphs_from_g6("dataset/positive_graphs.g6")
    graphs_train, graphs_val = train_test_split(positive_graphs, test_size=0.2)

    train_dataset = generate_paut_dataset(graphs_train, dataset_type="train")
    val_dataset = generate_paut_dataset(graphs_val, dataset_type="val")

    torch.save(train_dataset, "dataset/train_dataset.pt")
    torch.save(val_dataset, "dataset/val_dataset.pt")

    paut_sizes_to_csv(
        paut_sizes, f"dataset/partial_automorphism_sizes.csv")

    print(
        f"Generated {len(train_dataset)} train examples and {len(val_dataset)} val examples.")
