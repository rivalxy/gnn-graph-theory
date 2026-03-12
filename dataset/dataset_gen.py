import math
import os
import random
from collections import defaultdict
from typing import cast

import networkx as nx
import numpy as np
import torch
from pynauty import autgrp
from sklearn.model_selection import train_test_split
from sympy.combinatorics import Permutation, PermutationGroup
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from utils import (
    AdjacencyDict,
    DatasetType,
    GraphData,
    Mapping,
    PautStats,
    is_extensible,
    is_paut,
    paut_sizes_to_csv,
    read_graphs_from_g6,
)

MAX_ATTEMPTS = 100
MIN_PARTIAL_AUT_FRACTION = 0.5
MAX_PARTIAL_AUT_FRACTION = 0.8
MAX_BLOCKING_CANDIDATES = 5

# Feature indices for extra features
FEATURE_NODE_ID = 0
FEATURE_TARGET_ID = 1
FEATURE_SOURCE_ID = 2
FEATURE_DEGREE = 3
FEATURE_CLUSTERING = 4
FEATURE_TRIANGLES = 5
FEATURE_AVG_NEIGHBOR_DEGREE = 6


def is_identity_permutation(perm: list[int]) -> bool:
    return all(i == mapped for i, mapped in enumerate(perm))


def sample_partial_size(num_of_nodes: int, upper_bound: int | None = None) -> int:
    min_size = math.ceil(num_of_nodes * MIN_PARTIAL_AUT_FRACTION)
    max_size = (
        upper_bound
        if upper_bound is not None
        else math.floor(num_of_nodes * MAX_PARTIAL_AUT_FRACTION)
    )
    return random.randint(min_size, max_size)


def normalize(values: torch.Tensor) -> torch.Tensor:
    if values.numel() == 0:
        return values

    max_value = values.max().item()
    if max_value <= 0:
        return values

    return values / max_value


def gen_positive_examples(
    group: PermutationGroup, num_of_nodes: int, examples_num: int
) -> list[tuple[Mapping, int]]:
    seen_positives = set()
    positives = []
    attempts = 0
    nodes = list(range(num_of_nodes))

    while len(positives) < examples_num and attempts < MAX_ATTEMPTS * examples_num:
        attempts += 1
        perm = cast(Permutation, group.random()).array_form
        if is_identity_permutation(perm):
            continue

        p_aut_size = sample_partial_size(num_of_nodes)
        domain = random.sample(nodes, p_aut_size)
        mapping = {i: perm[i] for i in domain}

        key = frozenset(mapping.items())
        if key in seen_positives:
            continue

        seen_positives.add(key)
        positives.append((mapping, p_aut_size))
    return positives


def gen_negative_examples(
    group: PermutationGroup,
    examples_num: int,
    num_of_nodes: int,
    adjacency_list: AdjacencyDict,
) -> list[tuple[Mapping, int, int]]:
    """Generate negative examples (non-extensible partial automorphisms) using blocking strategy.

    Creates partial automorphisms that are locally valid but cannot be extended to full
    automorphisms by iteratively blocking their extensibility through strategic mapping additions.

    :param group: The automorphism group of the graph.
    :param examples_num: Number of negative examples to generate.
    :param num_of_nodes: Number of nodes in the graph.
    :param adjacency_list: Adjacency dictionary of the graph.
    :returns: List of tuples (mapping, original_paut_size, extension_size) where extension_size
        indicates how many nodes were added during the blocking process.
    """
    negatives = []
    seen_negatives = set()
    attempts = 0
    nodes = list(range(num_of_nodes))
    maximum_size = math.floor(num_of_nodes * MAX_PARTIAL_AUT_FRACTION)

    while len(negatives) < examples_num and attempts < MAX_ATTEMPTS * examples_num:
        attempts += 1
        perm = cast(Permutation, group.random()).array_form
        if is_identity_permutation(perm):
            continue

        p_aut_size = sample_partial_size(num_of_nodes, upper_bound=maximum_size) - 1
        original_paut_size = p_aut_size
        domain = random.sample(nodes, p_aut_size)
        mapping = {i: perm[i] for i in domain}

        maximum_extension = maximum_size - p_aut_size
        extension_size = random.randint(1, maximum_extension)
        current_extension = 0
        extension_attempts = 0

        while current_extension < extension_size and extension_attempts < MAX_ATTEMPTS:
            extension_attempts += 1
            new_mapping = block_automorphism(mapping, num_of_nodes, adjacency_list)

            if new_mapping is None:
                break

            mapping = new_mapping
            current_extension += 1

        if not is_paut(adjacency_list, mapping):
            continue
        if is_extensible(group, mapping):
            continue

        key = frozenset(mapping.items())
        if key in seen_negatives:
            continue

        seen_negatives.add(key)
        negatives.append((mapping, original_paut_size, current_extension))

    return negatives


def block_automorphism(
    positive: Mapping, num_of_nodes: int, adj: AdjacencyDict
) -> Mapping | None:
    nodes = list(range(num_of_nodes))

    unmapped_nodes = [n for n in nodes if n not in positive]
    targets = [n for n in nodes if n not in positive.values()]

    if not unmapped_nodes or not targets:
        return None

    random.shuffle(unmapped_nodes)
    random.shuffle(targets)

    for node in unmapped_nodes[: min(MAX_BLOCKING_CANDIDATES, len(unmapped_nodes))]:
        node_neighbors = adj.get(node, set())

        # TODO try only from other orbits, and with the same degree
        for target in targets[: min(MAX_BLOCKING_CANDIDATES, len(targets))]:
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


def make_pyg_data(
    tensor_edge_index: torch.Tensor,
    num_of_nodes: int,
    mapping: Mapping,
    label: int,
    extra_features: bool,
) -> Data:
    """Create a PyTorch Geometric Data object from a partial automorphism mapping.

    :param tensor_edge_index: Edge index tensor in PyG format.
    :param num_of_nodes: Number of nodes in the graph.
    :param mapping: Partial automorphism mapping.
    :param label: Binary label (1 for extensible, 0 for non-extensible).
    :param extra_features: Whether to include additional graph features.
    :returns: PyTorch Geometric Data object.
    """
    if extra_features:
        x = torch.full((num_of_nodes, 7), -1.0, dtype=torch.float)
        data = Data(edge_index=tensor_edge_index, num_nodes=num_of_nodes)
        nx_graph = to_networkx(data, to_undirected=True)

        # Add normalized degree as a feature
        degrees = torch.tensor(
            [nx_graph.degree(node) for node in range(num_of_nodes)], dtype=torch.float
        )
        x[:, FEATURE_DEGREE] = normalize(degrees)

        # Add clustering coefficient as a feature
        clustering_coeffs = torch.tensor(
            [nx.clustering(nx_graph, node) for node in range(num_of_nodes)],
            dtype=torch.float,
        )
        x[:, FEATURE_CLUSTERING] = clustering_coeffs

        # Add a normalized triangle count
        triangle_counts = torch.tensor(
            [nx.triangles(nx_graph, node) for node in range(num_of_nodes)],
            dtype=torch.float,
        )
        x[:, FEATURE_TRIANGLES] = normalize(triangle_counts)

        # Add average neighbor degree
        avg_neighbor_degrees = []
        for node in range(num_of_nodes):
            neighbors = list(nx_graph.neighbors(node))
            if neighbors:
                avg_degree = np.mean(
                    [nx_graph.degree(neighbor) for neighbor in neighbors]
                )
            else:
                avg_degree = 0.0
            avg_neighbor_degrees.append(avg_degree)

        avg_neighbor_degrees = torch.tensor(avg_neighbor_degrees, dtype=torch.float)
        x[:, FEATURE_AVG_NEIGHBOR_DEGREE] = normalize(avg_neighbor_degrees)

    else:
        # 3 features: node_id, target_id, source_id
        x = torch.full((num_of_nodes, 3), -1.0, dtype=torch.float)

    # Give EVERY node its own identity
    for node in range(num_of_nodes):
        x[node, FEATURE_NODE_ID] = float(node) / num_of_nodes

    # Mark mapped nodes with bidirectional info
    for source, target in mapping.items():
        x[source, FEATURE_TARGET_ID] = float(target) / num_of_nodes  # target_id
        x[target, FEATURE_SOURCE_ID] = float(source) / num_of_nodes  # source_id

    y = torch.tensor([label], dtype=torch.float)
    data = Data(x=x, edge_index=tensor_edge_index, y=y)
    return data


def build_edge_index(adjacency_dict: AdjacencyDict) -> torch.Tensor:
    """Build PyTorch Geometric edge index from an adjacency dictionary.

    :param adjacency_dict: Adjacency dictionary of the graph.
    :returns: Edge index tensor in PyG format (2 x num_edges).
    """
    if len(adjacency_dict) == 0:
        return torch.empty((2, 0), dtype=torch.long)

    edges = []
    for u, neighbors in adjacency_dict.items():
        for v in neighbors:
            edges.append([u, v])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index


def generate_paut_dataset(
    pynauty_graphs: list[GraphData], dataset_type: DatasetType, config: tuple[int, bool]
) -> tuple[list[Data], dict[int, list[PautStats]]]:
    """Generate partial automorphism dataset with labels from a list of pynauty graphs.

    :param pynauty_graphs: List of pynauty graphs.
    :param dataset_type: Type of the dataset (e.g., "train" or "val").
    :param config: Tuple of (max_examples_num, extra_features).
    :returns: List of PyG Data objects containing partial automorphism mappings and labels, and a dictionary of PautStats.
    """
    max_examples_num, extra_features = config

    positive_pyg_data = []
    negative_pyg_data = []
    paut_sizes = defaultdict(list)

    for graph_data in pynauty_graphs:
        pynauty_graph = graph_data.graph
        num_of_nodes = graph_data.num_of_nodes
        adjacency_dict = graph_data.adjacency_dict
        generators_raw, grpsize1, grpsize2, _, _ = autgrp(pynauty_graph)
        group_size = grpsize1 * 10**grpsize2

        # ensure up to 40 examples per graph with 1:1 ratio of positive to negative examples
        examples_num = int(min(max_examples_num, group_size))
        generators = [Permutation(g) for g in generators_raw]
        group = PermutationGroup(generators)

        tensor_edge_index = build_edge_index(adjacency_dict)

        positives = gen_positive_examples(group, num_of_nodes, examples_num)

        for mapping, p_aut_size in positives:
            assert is_paut(adjacency_dict, mapping)
            assert is_extensible(group, mapping)

            paut_sizes[num_of_nodes].append(PautStats(p_aut_size, 0, dataset_type))
            positive_pyg_data.append(
                make_pyg_data(
                    tensor_edge_index,
                    num_of_nodes,
                    mapping,
                    label=1,
                    extra_features=extra_features,
                )
            )

        negatives = gen_negative_examples(
            group, examples_num, num_of_nodes, adjacency_dict
        )

        for mapping, original_paut_size, extension_size in negatives:
            assert is_paut(adjacency_dict, mapping)
            assert not is_extensible(group, mapping)

            paut_sizes[num_of_nodes].append(
                PautStats(original_paut_size, extension_size, dataset_type)
            )
            negative_pyg_data.append(
                make_pyg_data(
                    tensor_edge_index,
                    num_of_nodes,
                    mapping,
                    label=0,
                    extra_features=extra_features,
                )
            )

    dataset = positive_pyg_data + negative_pyg_data
    return dataset, paut_sizes


if __name__ == "__main__":
    positive_graphs = read_graphs_from_g6("positive_graphs.g6")

    graphs_train, graphs_val = train_test_split(positive_graphs, test_size=0.2)

    configurations = {
        "baseline": (10, False),
        "7_features": (10, True),
        "larger": (20, False),
    }

    for config_name, config in configurations.items():
        print(f"Generating dataset for configuration: {config_name}")

        paut_sizes = defaultdict(list)

        train_dataset, train_paut_sizes = generate_paut_dataset(
            graphs_train, DatasetType.TRAIN, config=config
        )

        val_dataset, val_paut_sizes = generate_paut_dataset(
            graphs_val, DatasetType.VAL, config=config
        )

        paut_sizes = defaultdict(list)
        for node_count, stats in train_paut_sizes.items():
            paut_sizes[node_count].extend(stats)
        for node_count, stats in val_paut_sizes.items():
            paut_sizes[node_count].extend(stats)

        if config_name == "larger":
            torch.save(train_dataset, "train_dataset.pt")
            torch.save(val_dataset, "val_dataset.pt")

            paut_sizes_to_csv(paut_sizes, "paut_sizes.csv")
        else:
            os.makedirs(config_name, exist_ok=True)
            torch.save(train_dataset, f"{config_name}/train_dataset_{config_name}.pt")
            torch.save(val_dataset, f"{config_name}/val_dataset_{config_name}.pt")

            paut_sizes_to_csv(paut_sizes, f"{config_name}/paut_sizes_{config_name}.csv")

        print(
            f"Generated {len(train_dataset)} train examples and {len(val_dataset)} val examples."
        )
