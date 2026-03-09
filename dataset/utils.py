import csv
from dataclasses import dataclass
from enum import StrEnum
from typing import TypeAlias

import networkx as nx
import pynauty
from sympy.combinatorics import PermutationGroup

Edge: TypeAlias = tuple[int, int]
Mapping: TypeAlias = dict[int, int]
AdjacencyDict: TypeAlias = dict[int, set[int]]


def build_adjacency_dict(edge_list: list[Edge]) -> AdjacencyDict:
    """Build an adjacency dictionary from a list of edges.

    :param edge_list: List of edges as (u, v) tuples.
    :returns: Dictionary mapping each node to its set of neighbors.
    """
    adjacency_dict: AdjacencyDict = {}
    for u, v in edge_list:
        adjacency_dict.setdefault(u, set()).add(v)
        adjacency_dict.setdefault(v, set()).add(u)
    return adjacency_dict


@dataclass
class GraphData:
    graph: pynauty.Graph
    num_of_nodes: int
    adjacency_dict: AdjacencyDict


def read_graphs_from_g6(file_path: str) -> list[GraphData]:
    """Read graphs from a .g6 file and convert them to pynauty format.

    :param file_path: Path to the .g6 file.
    :returns: List of GraphData objects containing the pynauty graph, number of nodes, and adjacency dictionary.
    """
    graphs: list[nx.Graph] = nx.read_graph6(file_path)
    graph_data_list = []
    for graph in graphs:
        num_of_nodes = graph.number_of_nodes()
        adjacency_dict = build_adjacency_dict(graph.edges())
        pynauty_graph = pynauty.Graph(num_of_nodes)
        pynauty_graph.set_adjacency_dict(adjacency_dict)
        graph_data_list.append(GraphData(pynauty_graph, num_of_nodes, adjacency_dict))
    return graph_data_list


def is_injective(mapping: Mapping) -> bool:
    """Check if the mapping is injective (one-to-one).

    :param mapping: A partial mapping from node indices to node indices.
    :returns: True if the mapping is injective, False otherwise.
    """
    return len(set(mapping.values())) == len(mapping)


def is_paut(adjacency_dict: AdjacencyDict, mapping: Mapping) -> bool:
    """Check if mapping is a partial automorphism on given graph.

    :param adjacency_dict: Adjacency dictionary of the graph.
    :param mapping: A partial mapping from node indices to node indices.
    :returns: True if the mapping is a partial automorphism, False otherwise.
    """
    if not mapping:
        return False

    if not is_injective(mapping):
        return False

    domain = list(mapping.keys())
    for i, u in enumerate(domain):
        for v in domain[i + 1 :]:
            u_mapped = mapping[u]
            v_mapped = mapping[v]
            if (v in adjacency_dict.get(u, set())) != (
                v_mapped in adjacency_dict.get(u_mapped, set())
            ):
                return False
    return True


def is_extensible(group: PermutationGroup, mapping: Mapping) -> bool:
    """Check if mapping can be extended to a full automorphism on given graph.

    :param group: The automorphism group of the graph.
    :param mapping: A partial mapping from node indices to node indices.
        Indices must be contiguous non-negative integers matching the pynauty graph representation.
    :returns: True if the mapping can be extended to a full automorphism, False otherwise.
    """
    domain = set(mapping.keys())
    for perm in group.generate():
        if all(perm.array_form[i] == mapping[i] for i in domain):
            return True
    return False


class DatasetType(StrEnum):
    TRAIN = "train"
    VAL = "val"


@dataclass
class PautStats:
    original_paut_size: int
    extension_size: int
    dataset_type: DatasetType


def paut_sizes_to_csv(
    stats_by_node_count: dict[int, list[PautStats]], file_path: str
) -> None:
    """
    Writes PautStats grouped by node count to a CSV file.

    :param stats_by_node_count: Dictionary mapping number of nodes to a list of PautStats.
    :param file_path: Path to the output CSV file.
    """
    with open(file_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            ["num_of_nodes", "original_paut_size", "extension_size", "dataset_type"]
        )
        for num_of_nodes, stats in stats_by_node_count.items():
            for stat in stats:
                writer.writerow(
                    [
                        num_of_nodes,
                        stat.original_paut_size,
                        stat.extension_size,
                        stat.dataset_type,
                    ]
                )
