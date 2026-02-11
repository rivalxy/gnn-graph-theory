import networkx as nx
import csv

from pynauty import Graph
from sympy.combinatorics import Permutation, PermutationGroup
from pynauty import Graph, autgrp
from collections import defaultdict


def build_adjacency_list(edge_list: list[tuple]) -> defaultdict[int, set]:
    adjacency_list = defaultdict(set)
    for u, v in edge_list:
        adjacency_list[u].add(v)
        adjacency_list[v].add(u)
    return adjacency_list


def read_graphs_from_g6(file_path: str) -> list[Graph]:
    """
    Reads graphs from a .g6 file and converts them to pynauty format.

    :param file_path: Path to the .g6 file.
    :returns: List of pynauty graphs with their number of nodes and edges.
    """

    graphs = nx.read_graph6(file_path)
    pynauty_graphs = []
    for graph in graphs:
        num_of_nodes = int(graph.number_of_nodes())
        adjacency_list = build_adjacency_list(graph.edges())
        pynauty_graph = Graph(num_of_nodes)
        pynauty_graph.set_adjacency_dict(adjacency_list)
        pynauty_graphs.append(
            (pynauty_graph, num_of_nodes, adjacency_list))
    return pynauty_graphs


def is_paut(adjacency_list: dict[int, set], mapping: dict[int, int]) -> bool:
    """
    Check if mapping is a partial automorphism on given graph.
    """

    # Check injectivity
    if len(set(mapping.values())) != len(mapping):
        return False

    domain = list(mapping.keys())
    for i, u in enumerate(domain):
        for v in domain[i+1:]:
            u_mapped = mapping[u]
            v_mapped = mapping[v]
            if (v in adjacency_list.get(u, set())) != (v_mapped in adjacency_list.get(u_mapped, set())):
                return False
    return True


def is_extensible(group: PermutationGroup, mapping: dict[int, int]) -> bool:
    """
    Check if mapping can be extended to a full automorphism on given graph.
    """
    domain = set(mapping.keys())
    for perm in group.generate():
        if all(perm.array_form[i] == mapping[i] for i in domain):
            return True
    return False


def paut_sizes_to_csv(examples: dict[int, list[tuple]], file_path: str):
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['num_of_nodes', 'original_paut_size', 'extension_size', 'dataset_type'])
        for num_of_nodes, stats in examples.items():
            for original_paut_size, extension_size, example_type in stats:
                writer.writerow([num_of_nodes, original_paut_size, extension_size, example_type])


if __name__ == "__main__":
    test_graph = nx.Graph()
    test_graph.add_edges_from(
        [(0, 1), (1, 2), (2, 3), (3, 4), (2, 4), (4, 5), (5, 6)])
    test_adjacency_list = build_adjacency_list(test_graph.edges())
    positive_mappings = [{0: 0, 1: 1, 2: 2},
                         {0: 0, 1: 1, 4: 4}]
    negative_mappings = [{0: 2, 1: 1, 2: 0},
                         {0: 0, 1: 1, 2: 2, 3: 4, 4: 3},
                         ]

    num_of_nodes = int(test_graph.number_of_nodes())
    pynauty_graph = Graph(num_of_nodes)
    pynauty_graph.set_adjacency_dict(test_adjacency_list)

    generators_raw, grpsize1, grpsize2, _, _ = autgrp(pynauty_graph)
    group_size = grpsize1 * 10**grpsize2
    generators = [Permutation(g) for g in generators_raw]
    group = PermutationGroup(generators)

    for mapping in positive_mappings:
        assert is_paut(test_adjacency_list, mapping)
        assert is_extensible(group, mapping)
    for mapping in negative_mappings:
        assert is_paut(test_adjacency_list, mapping)
        assert not is_extensible(group, mapping)
    print("All tests passed.")
