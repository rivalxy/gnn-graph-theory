import networkx as nx
from pynauty import Graph

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


def is_paut() -> bool:
    """
    Check if mapping is a partial automorphism on given graph.
    """
    return False


def is_extensible() -> bool:
    """
    Check if mapping can be extended to a full automorphism on given graph.
    """
    return False
