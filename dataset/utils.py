import networkx as nx
from pynauty import Graph
from sympy.combinatorics import Permutation, PermutationGroup
from pynauty import Graph, autgrp


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
        pynauty_graph = Graph(num_of_nodes)
        pynauty_graph.set_adjacency_dict(dict(graph.adjacency()))
        pynauty_graphs.append(
            (pynauty_graph, num_of_nodes, graph.edges()))
    return pynauty_graphs


def is_paut(edge_list: list[tuple], mapping: dict[int, int]) -> bool:
    """
    Check if mapping is a partial automorphism on given graph.
    """

    edge_set = set()
    for u, v in edge_list:
        edge_set.add((u, v))
        edge_set.add((v, u))

    domain = list(mapping.keys())
    for i, u in enumerate(domain):
        for v in domain[i+1:]:
            u_mapped = mapping[u]
            v_mapped = mapping[v]
            if ((u, v) in edge_set) != ((u_mapped, v_mapped) in edge_set):
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


if __name__ == "__main__":
    test_graph = nx.Graph()
    test_graph.add_edges_from(
        [(0, 1), (1, 2), (2, 3), (3, 4), (2, 4), (4, 5), (5, 6)])
    test_edge_list = list(test_graph.edges())
    positive_mappings = [{0: 0, 1: 1, 2: 2},
                         {0: 0, 1: 1, 4: 4}]
    negative_mappings = [{0: 2, 1: 1, 2: 0},
                         {0: 0, 1: 1, 2: 2, 3: 4, 4: 3},
                         ]

    num_of_nodes = int(test_graph.number_of_nodes())
    pynauty_graph = Graph(num_of_nodes)
    pynauty_graph.set_adjacency_dict(dict(test_graph.adjacency()))
    generators_raw, grpsize1, grpsize2, _, _ = autgrp(pynauty_graph)
    group_size = grpsize1 * 10**grpsize2
    generators = [Permutation(g) for g in generators_raw]
    group = PermutationGroup(generators)

    for mapping in positive_mappings:
        assert is_paut(test_edge_list, mapping)
        assert is_extensible(group, mapping)
    for mapping in negative_mappings:
        assert is_paut(test_edge_list, mapping)
        assert not is_extensible(group, mapping)
    print("All tests passed.")
