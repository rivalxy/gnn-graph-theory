from collections import defaultdict
from dataclasses import dataclass

import torch
from pynauty import autgrp
from sympy.combinatorics import Permutation, PermutationGroup
from torch_geometric.data import Data

from dataset.features import make_pyg_data
from dataset.graph_utils import (
    AdjacencyDict,
    DatasetType,
    GraphData,
    Mapping,
    PautStats,
    is_extensible,
    is_paut,
)
from dataset.sampling import (
    gen_blocking_examples,
    gen_positive_examples,
    gen_pseudo_similar_examples,
    mapping_key,
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
    """Build a PyG edge index from an adjacency dictionary."""
    if len(adjacency_dict) == 0:
        return torch.empty((2, 0), dtype=torch.long)

    edges = []
    for u, neighbors in adjacency_dict.items():
        for v in neighbors:
            edges.append([u, v])
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


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


def generate_raw_examples(
    pynauty_graphs: list[GraphData],
    dataset_type: DatasetType,
    max_examples_num: int,
) -> list[RawPautExample]:
    """Generate raw partial automorphism examples without feature encoding."""
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
            sample
            for sample in negatives_blocking
            if mapping_key(sample[0]) not in negative_seen_keys
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
    """Convert raw examples to PyG Data objects for one feature variant."""
    pyg_data = []
    paut_sizes: dict[int, list[PautStats]] = defaultdict(list)

    for example in raw_examples:
        paut_sizes[example.num_of_nodes].append(example.paut_stats)
        pyg_data.append(
            make_pyg_data(
                example.edge_index,
                example.num_of_nodes,
                example.mapping,
                label=example.label,
                extra_features=extra_features,
            )
        )

    return pyg_data, paut_sizes


def generate_paut_dataset(
    pynauty_graphs: list[GraphData], dataset_type: DatasetType, config: tuple[int, bool]
) -> tuple[list[Data], dict[int, list[PautStats]]]:
    """Generate a labeled partial automorphism dataset from pynauty graphs."""
    max_examples_num, extra_features = config
    raw_examples = generate_raw_examples(pynauty_graphs, dataset_type, max_examples_num)
    return raw_examples_to_pyg(raw_examples, extra_features)
