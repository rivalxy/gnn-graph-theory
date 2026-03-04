from dataclasses import dataclass
from enum import StrEnum
import networkx as nx
import csv
import pynauty

from sympy.combinatorics import PermutationGroup
from collections import defaultdict


def _build_adjacency_dict(edge_list: list[tuple]) -> defaultdict[int, set]:
    adjacency_dict = defaultdict(set)
    for u, v in edge_list:
        adjacency_dict[u].add(v)
        adjacency_dict[v].add(u)
    return adjacency_dict


@dataclass
class GraphData:
    graph: pynauty.Graph
    num_of_nodes: int
    adjacency_dict: dict[int, set]


def read_graphs_from_g6(file_path: str) -> list[GraphData]:
    """
    Reads graphs from a .g6 file and converts them to pynauty format.

    :param file_path: Path to the .g6 file.
    :returns: List of GraphData objects containing the pynauty graph, number of nodes, and adjacency dictionary.
    """

    graphs: list[nx.Graph] = nx.read_graph6(file_path)
    pynauty_graphs = []
    for graph in graphs:
        num_of_nodes = graph.number_of_nodes()
        adjacency_dict = _build_adjacency_dict(graph.edges())
        pynauty_graph = pynauty.Graph(num_of_nodes)
        pynauty_graph.set_adjacency_dict(adjacency_dict)
        pynauty_graphs.append(
            GraphData(pynauty_graph, num_of_nodes, adjacency_dict))
    return pynauty_graphs


def is_paut(adjacency_dict: dict[int, set], mapping: dict[int, int]) -> bool:
    """
    Check if mapping is a partial automorphism on given graph.

    :param adjacency_dict: Adjacency dictionary of the graph.
    :param mapping: A partial mapping from node indices to node indices.
    :returns: True if the mapping is a partial automorphism, False otherwise.
    """

    # Check injectivity
    if len(set(mapping.values())) != len(mapping):
        return False

    domain = list(mapping.keys())
    for i, u in enumerate(domain):
        for v in domain[i+1:]:
            u_mapped = mapping[u]
            v_mapped = mapping[v]
            if (v in adjacency_dict.get(u, set())) != (v_mapped in adjacency_dict.get(u_mapped, set())):
                return False
    return True


def is_extensible(group: PermutationGroup, mapping: dict[int, int]) -> bool:
    """
    Check if mapping can be extended to a full automorphism on given graph.

    :param group: The automorphism group of the graph.
    :param mapping: A partial mapping from node indices to node indices.
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


def paut_sizes_to_csv(stats_by_node_count: dict[int, list[PautStats]], file_path: str) -> None:
    """
    Writes PautStats grouped by node count to a CSV file.

    :param stats_by_node_count: Dictionary mapping number of nodes to a list of PautStats.
    :param file_path: Path to the output CSV file.
    """
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['num_of_nodes', 'original_paut_size',
                        'extension_size', 'dataset_type'])
        for num_of_nodes, stats in stats_by_node_count.items():
            for stat in stats:
                writer.writerow(
                    [num_of_nodes, stat.original_paut_size, stat.extension_size, stat.dataset_type])
