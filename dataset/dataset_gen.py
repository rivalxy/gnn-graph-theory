import json
import math
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import cast

import networkx as nx
import torch
from data_utils import (
    AdjacencyDict,
    DatasetType,
    GraphData,
    Mapping,
    PautStats,
    bfs_expand_pseudo_similar,
    build_orbit_map,
    find_pseudo_similar_pair,
    is_extensible,
    is_paut,
    paut_sizes_to_csv,
    read_graphs_from_g6,
)
from pynauty import autgrp
from sklearn.model_selection import train_test_split
from sympy.combinatorics import Permutation, PermutationGroup
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

MAX_ATTEMPTS = 100
MIN_PARTIAL_AUT_FRACTION = 0.5
MAX_PARTIAL_AUT_FRACTION = 0.8
MAX_BLOCKING_CANDIDATES = 5
BASELINE_FEATURE_DIM = 3
EXTRA_FEATURE_DIM = 7

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


def mapping_key(mapping: Mapping) -> frozenset[tuple[int, int]]:
    return frozenset(mapping.items())


def partial_size_bounds(
    num_of_nodes: int, upper_bound: int | None = None
) -> tuple[int, int]:
    min_size = math.ceil(num_of_nodes * MIN_PARTIAL_AUT_FRACTION)
    max_size = (
        upper_bound
        if upper_bound is not None
        else math.floor(num_of_nodes * MAX_PARTIAL_AUT_FRACTION)
    )
    return min_size, max_size


def sample_partial_size(num_of_nodes: int, upper_bound: int | None = None) -> int:
    min_size, max_size = partial_size_bounds(num_of_nodes, upper_bound)
    return random.randint(min_size, max_size)


def normalize(values: torch.Tensor) -> torch.Tensor:
    if values.numel() == 0:
        return values

    max_value = values.max().item()
    if max_value <= 0:
        return values

    return values / max_value


def is_non_extensible_paut(
    adjacency_list: AdjacencyDict,
    group: PermutationGroup,
    mapping: Mapping,
) -> bool:
    return is_paut(adjacency_list, mapping) and not is_extensible(group, mapping)


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

        key = mapping_key(mapping)
        if key in seen_positives:
            continue

        seen_positives.add(key)
        positives.append((mapping, len(mapping)))
    return positives


def gen_pseudo_similar_examples(
    group: PermutationGroup,
    num_of_nodes: int,
    adjacency_list: AdjacencyDict,
    examples_num: int,
) -> list[tuple[Mapping, int]]:
    """Generate hard negative examples using pseudo-similar vertex pairs.

    A pseudo-similar pair (u, v) satisfies two properties simultaneously:
      1. G - u ≅ G - v  — so the seed {u → v} passes all local checks
      2. u and v are in different orbits of Aut(G)  — so {u → v} is
         provably non-extendable, without needing any blocking step

    The seed is then grown by BFS using the witnessing isomorphism as a
    guide, until the mapping covers between MIN and MAX partial-aut fraction
    of the graph.

    Returns (mapping, size) pairs where size is the final mapping size.

    :param group: Automorphism group of the graph.
    :param num_of_nodes: Number of nodes in the graph.
    :param adjacency_list: Adjacency dictionary of the graph.
    :param examples_num: Target number of negatives to produce.
    :returns: List of (mapping, size,) tuples. Empty if the graph has no
        pseudo-similar pair (method degrades gracefully).
    """
    pair = find_pseudo_similar_pair(adjacency_list, group, num_of_nodes)
    if pair is None:
        return []

    u, v, sigma = pair
    negatives: list[tuple[Mapping, int]] = []
    seen: set[frozenset[tuple[int, int]]] = set()

    min_size, max_size = partial_size_bounds(num_of_nodes)
    attempts = 0

    while len(negatives) < examples_num and attempts < MAX_ATTEMPTS * examples_num:
        attempts += 1
        target_size = random.randint(min_size, max_size)

        mapping = bfs_expand_pseudo_similar(adjacency_list, u, v, sigma, target_size)

        if len(mapping) < min_size:
            continue
        if not is_non_extensible_paut(adjacency_list, group, mapping):
            continue

        key = mapping_key(mapping)
        if key in seen:
            continue
        seen.add(key)
        negatives.append((mapping, len(mapping)))

    return negatives


def gen_blocking_examples(
    group: PermutationGroup,
    examples_num: int,
    num_of_nodes: int,
    adjacency_list: AdjacencyDict,
) -> list[tuple[Mapping, int]]:
    """Generate negative examples (non-extensible partial automorphisms) using blocking strategy.

    Creates partial automorphisms that are locally valid but cannot be extended to full
    automorphisms by iteratively blocking their extensibility through strategic mapping additions.

    :param group: The automorphism group of the graph.
    :param examples_num: Number of negative examples to generate.
    :param num_of_nodes: Number of nodes in the graph.
    :param adjacency_list: Adjacency dictionary of the graph.
    :returns: List of tuples (mapping, size) where size indicates the number of nodes in the partial automorphism.
    """
    negatives: list[tuple[Mapping, int]] = []
    seen_negatives: set[frozenset[tuple[int, int]]] = set()
    attempts = 0
    nodes = list(range(num_of_nodes))
    maximum_size = math.floor(num_of_nodes * MAX_PARTIAL_AUT_FRACTION)

    while len(negatives) < examples_num and attempts < MAX_ATTEMPTS * examples_num:
        attempts += 1
        perm = cast(Permutation, group.random()).array_form
        if is_identity_permutation(perm):
            continue

        p_aut_size = sample_partial_size(num_of_nodes, upper_bound=maximum_size) - 1
        domain = random.sample(nodes, p_aut_size)
        mapping = {i: perm[i] for i in domain}

        maximum_extension = maximum_size - p_aut_size
        extension_size = random.randint(1, maximum_extension)
        current_extension = 0
        extension_attempts = 0

        while current_extension < extension_size and extension_attempts < MAX_ATTEMPTS:
            extension_attempts += 1
            new_mapping = block_automorphism(
                mapping, num_of_nodes, adjacency_list, group
            )

            if new_mapping is None:
                break

            mapping = new_mapping
            current_extension += 1

        if not is_non_extensible_paut(adjacency_list, group, mapping):
            continue

        key = mapping_key(mapping)
        if key in seen_negatives:
            continue

        seen_negatives.add(key)
        negatives.append((mapping, len(mapping)))

    return negatives


def block_automorphism(
    positive: Mapping, num_of_nodes: int, adj: AdjacencyDict, group: PermutationGroup
) -> Mapping | None:
    nodes = list(range(num_of_nodes))
    orbit_of = build_orbit_map(group)

    unmapped_nodes = [n for n in nodes if n not in positive]
    targets = [n for n in nodes if n not in positive.values()]

    if not unmapped_nodes or not targets:
        return None

    random.shuffle(unmapped_nodes)
    random.shuffle(targets)

    for node in unmapped_nodes[: min(MAX_BLOCKING_CANDIDATES, len(unmapped_nodes))]:
        non_orbit_targets = [
            t for t in targets if orbit_of.get(t) != orbit_of.get(node)
        ]

        for target in non_orbit_targets[
            : min(MAX_BLOCKING_CANDIDATES, len(non_orbit_targets))
        ]:
            test_map = positive.copy()
            test_map[node] = target

            if is_non_extensible_paut(adj, group, test_map):
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
        x = build_extra_feature_matrix(tensor_edge_index, num_of_nodes)
    else:
        # 3 features: node_id, target_id, source_id
        x = torch.full((num_of_nodes, BASELINE_FEATURE_DIM), -1.0, dtype=torch.float)

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


def build_extra_feature_matrix(
    tensor_edge_index: torch.Tensor, num_of_nodes: int
) -> torch.Tensor:
    x = torch.full((num_of_nodes, EXTRA_FEATURE_DIM), -1.0, dtype=torch.float)
    pyg_graph = Data(edge_index=tensor_edge_index, num_nodes=num_of_nodes)
    nx_graph = to_networkx(pyg_graph, to_undirected=True)

    degrees = torch.tensor(
        [nx_graph.degree(node) for node in range(num_of_nodes)], dtype=torch.float
    )
    x[:, FEATURE_DEGREE] = normalize(degrees)

    clustering_coeffs = torch.tensor(
        [nx.clustering(nx_graph, node) for node in range(num_of_nodes)],
        dtype=torch.float,
    )
    x[:, FEATURE_CLUSTERING] = clustering_coeffs

    triangle_counts = torch.tensor(
        [nx.triangles(nx_graph, node) for node in range(num_of_nodes)],
        dtype=torch.float,
    )
    x[:, FEATURE_TRIANGLES] = normalize(triangle_counts)

    avg_neighbor_degrees = []
    for node in range(num_of_nodes):
        neighbors = list(nx_graph.neighbors(node))
        if neighbors:
            avg_degree = sum(nx_graph.degree(neighbor) for neighbor in neighbors) / len(
                neighbors
            )
        else:
            avg_degree = 0.0
        avg_neighbor_degrees.append(avg_degree)

    x[:, FEATURE_AVG_NEIGHBOR_DEGREE] = normalize(
        torch.tensor(avg_neighbor_degrees, dtype=torch.float)
    )
    return x


def append_validated_examples(
    raw_examples: list[RawPautExample],
    examples: list[tuple[Mapping, int]],
    *,
    edge_index: torch.Tensor,
    num_of_nodes: int,
    adjacency_dict: AdjacencyDict,
    group: PermutationGroup,
    label: int,
    dataset_type: DatasetType,
) -> None:
    expected_extensible = label == 1
    for mapping, p_aut_size in examples:
        assert is_paut(adjacency_dict, mapping)
        assert is_extensible(group, mapping) == expected_extensible

        raw_examples.append(
            RawPautExample(
                edge_index=edge_index,
                num_of_nodes=num_of_nodes,
                mapping=mapping,
                label=label,
                paut_stats=PautStats(p_aut_size, label, dataset_type),
            )
        )


@dataclass
class RawPautExample:
    """A raw partial automorphism example before feature encoding."""

    edge_index: torch.Tensor
    num_of_nodes: int
    mapping: Mapping
    label: int
    paut_stats: PautStats


@dataclass(frozen=True)
class DatasetConfiguration:
    """Configuration for encoding and saving one dataset variant."""

    name: str
    raw_train: list[RawPautExample]
    extra_features: bool
    val_paut_sizes: dict[int, list[PautStats]]
    train_output_path: str
    paut_sizes_output_path: str


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


def generate_raw_examples(
    pynauty_graphs: list[GraphData],
    dataset_type: DatasetType,
    max_examples_num: int,
) -> list[RawPautExample]:
    """Generate raw partial automorphism examples (mappings + labels) without feature encoding.

    :param pynauty_graphs: List of pynauty graphs.
    :param dataset_type: Type of the dataset (e.g., "train" or "val").
    :param max_examples_num: Maximum number of examples per graph.
    :returns: List of RawPautExample objects.
    """
    raw_examples: list[RawPautExample] = []

    for graph_data in pynauty_graphs:
        pynauty_graph = graph_data.graph
        num_of_nodes = graph_data.num_of_nodes
        adjacency_dict = graph_data.adjacency_dict
        generators_raw, grpsize1, grpsize2, _, _ = autgrp(pynauty_graph)
        group_size = int(round(grpsize1 * 10**grpsize2))

        examples_num = min(max_examples_num, group_size)
        generators = [Permutation(g) for g in generators_raw]
        group = PermutationGroup(generators)

        tensor_edge_index = build_edge_index(adjacency_dict)

        positives = gen_positive_examples(group, num_of_nodes, examples_num)

        append_validated_examples(
            raw_examples,
            positives,
            edge_index=tensor_edge_index,
            num_of_nodes=num_of_nodes,
            adjacency_dict=adjacency_dict,
            group=group,
            label=1,
            dataset_type=dataset_type,
        )

        negatives_pseudo = gen_pseudo_similar_examples(
            group, num_of_nodes, adjacency_dict, examples_num
        )

        negatives_blocking = gen_blocking_examples(
            group, examples_num, num_of_nodes, adjacency_dict
        )

        negative_seen_keys: set[frozenset[tuple[int, int]]] = {
            mapping_key(mapping) for mapping, _ in negatives_pseudo
        }
        negatives_blocking_filtered = [
            t for t in negatives_blocking if mapping_key(t[0]) not in negative_seen_keys
        ]
        negatives = (negatives_pseudo + negatives_blocking_filtered)[: len(positives)]

        append_validated_examples(
            raw_examples,
            negatives,
            edge_index=tensor_edge_index,
            num_of_nodes=num_of_nodes,
            adjacency_dict=adjacency_dict,
            group=group,
            label=0,
            dataset_type=dataset_type,
        )

    return raw_examples


def raw_examples_to_pyg(
    raw_examples: list[RawPautExample], extra_features: bool
) -> tuple[list[Data], dict[int, list[PautStats]]]:
    """Convert raw examples to PyG Data objects with the given feature setting.

    :param raw_examples: List of RawPautExample objects.
    :param extra_features: Whether to include additional graph features.
    :returns: List of PyG Data objects and a dictionary of PautStats.
    """
    pyg_data = []
    paut_sizes: dict[int, list[PautStats]] = defaultdict(list)

    for ex in raw_examples:
        paut_sizes[ex.num_of_nodes].append(ex.paut_stats)
        pyg_data.append(
            make_pyg_data(
                ex.edge_index,
                ex.num_of_nodes,
                ex.mapping,
                label=ex.label,
                extra_features=extra_features,
            )
        )

    return pyg_data, paut_sizes


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
    raw_examples = generate_raw_examples(pynauty_graphs, dataset_type, max_examples_num)
    return raw_examples_to_pyg(raw_examples, extra_features)


if __name__ == "__main__":
    positive_graphs = read_graphs_from_g6("all_graphs.g6")

    graphs_train, graphs_temp, idx_train, idx_temp = train_test_split(
        positive_graphs, range(len(positive_graphs)), test_size=0.2, random_state=42
    )
    graphs_val, graphs_test, idx_val, idx_test = train_test_split(
        graphs_temp, idx_temp, test_size=0.5, random_state=42
    )

    splits = {
        "train": list(idx_train),
        "val": list(idx_val),
        "test": list(idx_test),
    }
    with open("splits.json", "w") as f:
        json.dump(splits, f)

    # Generate val and test examples once (shared across all configurations)
    VAL_TEST_MAX_EXAMPLES = 10
    print("Generating shared val examples...")
    raw_val = generate_raw_examples(graphs_val, DatasetType.VAL, VAL_TEST_MAX_EXAMPLES)
    print(f"Generated {len(raw_val)} raw val examples.")

    print("Generating shared test examples...")
    raw_test = generate_raw_examples(
        graphs_test, DatasetType.TEST, VAL_TEST_MAX_EXAMPLES
    )
    print(f"Generated {len(raw_test)} raw test examples.")

    # Encode val/test once per feature variant (baseline=3 features, 7_features=7 features)
    print("Encoding val/test (baseline)...")
    val_dataset_baseline, val_paut_sizes_baseline = raw_examples_to_pyg(
        raw_val, extra_features=False
    )
    test_dataset_baseline, _ = raw_examples_to_pyg(raw_test, extra_features=False)

    print("Encoding val/test (7_features)...")
    val_dataset_7f, val_paut_sizes_7f = raw_examples_to_pyg(
        raw_val, extra_features=True
    )
    test_dataset_7f, _ = raw_examples_to_pyg(raw_test, extra_features=True)

    # Generate raw train examples once per distinct max_examples_num.
    # baseline and 7_features share max_examples_num=10, larger uses 20.
    print("Generating train examples (max_examples=10)...")
    raw_train_10 = generate_raw_examples(graphs_train, DatasetType.TRAIN, 10)
    print(f"  train: {len(raw_train_10)}")

    print("Generating train examples (max_examples=20)...")
    raw_train_20 = generate_raw_examples(graphs_train, DatasetType.TRAIN, 20)
    print(f"  train: {len(raw_train_20)}")

    os.makedirs("baseline", exist_ok=True)
    os.makedirs("7_features", exist_ok=True)

    torch.save(val_dataset_baseline, "val_dataset.pt")
    torch.save(test_dataset_baseline, "test_dataset.pt")

    torch.save(val_dataset_7f, "7_features/val_dataset_7_features.pt")
    torch.save(test_dataset_7f, "7_features/test_dataset_7_features.pt")

    configurations = [
        DatasetConfiguration(
            name="baseline",
            raw_train=raw_train_10,
            extra_features=False,
            val_paut_sizes=val_paut_sizes_baseline,
            train_output_path="baseline/train_dataset_baseline.pt",
            paut_sizes_output_path="baseline/paut_sizes_baseline.csv",
        ),
        DatasetConfiguration(
            name="7_features",
            raw_train=raw_train_10,
            extra_features=True,
            val_paut_sizes=val_paut_sizes_7f,
            train_output_path="7_features/train_dataset_7_features.pt",
            paut_sizes_output_path="7_features/paut_sizes_7_features.csv",
        ),
        DatasetConfiguration(
            name="larger",
            raw_train=raw_train_20,
            extra_features=False,
            val_paut_sizes=val_paut_sizes_baseline,
            train_output_path="train_dataset.pt",
            paut_sizes_output_path="paut_sizes.csv",
        ),
    ]

    for config in configurations:
        print(f"Encoding train dataset for configuration: {config.name}")

        train_dataset, train_paut_sizes = raw_examples_to_pyg(
            config.raw_train, config.extra_features
        )

        paut_sizes: dict[int, list[PautStats]] = defaultdict(list)
        for node_count, stats in train_paut_sizes.items():
            paut_sizes[node_count].extend(stats)
        for node_count, stats in config.val_paut_sizes.items():
            paut_sizes[node_count].extend(stats)

        torch.save(train_dataset, config.train_output_path)
        paut_sizes_to_csv(paut_sizes, config.paut_sizes_output_path)

        print(
            f"Generated {len(train_dataset)} train, {len(val_dataset_baseline)} val, {len(test_dataset_baseline)} test examples."
        )
