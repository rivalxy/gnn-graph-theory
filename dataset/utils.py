import networkx as nx
from pynauty import Graph
from sympy.combinatorics import Permutation, PermutationGroup


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
        pynauty_graphs.append((pynauty_graph, num_of_nodes, graph.edges()))
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
