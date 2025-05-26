import networkx as nx
from igraph import Graph
from sklearn.model_selection import train_test_split
import random
import torch
from torch_geometric.data import Data


MAX_EXAMPLES_NUM = 30
MAX_ATTEMPTS = 100


def read_graphs_from_g6(file_path: str) -> list[Graph]:
    """
    Reads graphs from a .g6 file and converts them to igraph format.

    :param file_path: Path to the .g6 file.
    :returns: List of igraph graphs.
    """

    graphs = nx.read_graph6(file_path)
    igraphs = []
    for g in graphs:
        igraphs.append([Graph.from_networkx(g), g.number_of_nodes()])
    return igraphs


def generate_partial_automorphism_graphs(graphs: list[Graph]) -> list:
    """
    Generates partial automorphism graphs from a list of igraph graphs.

    :param graphs: List of igraph graphs.
    :returns: TODO
    """

    dataset = []

    for G, n in graphs:
        aut_group = G.get_automorphisms_vf2()
        aut_num = G.count_automorphisms_vf2()

        # ensure 10-30 examples per graph with 1:1 ratio of positive to negative examples
        examples_num = int(min(MAX_EXAMPLES_NUM, aut_num))
        # positive examples

        # negative examples
        
    return dataset


def _make_data(G: Graph, n: int,  mapping: dict[int, int], label: int) -> Data:
    x = torch.zeros((n, 2), dtype=torch.float)

    for u, v in mapping.items():
        x[u, 0] = 1.0
        x[v, 1] = 1.0

    edge_list = G.get_edgelist()
    if len(edge_list) > 0:
        edges = []
        for u, v in edge_list:
            edges.append([u, v])
            edges.append([v, u])
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    y = torch.tensor([label], dtype=torch.float)
    data = Data(x=x, edge_index=edge_index, y=y)
    data.mapping = mapping.copy()

    return data


if __name__ == "__main__":
    all_graphs = read_graphs_from_g6("dataset/2000_raw_graphs.g6")
    graphs_train, graphs_val = train_test_split(all_graphs, test_size=0.2)
    train_dataset = generate_partial_automorphism_graphs(graphs_train)
    val_dataset = generate_partial_automorphism_graphs(graphs_val)
    print(len(all_graphs), len(train_dataset), len(val_dataset))
    
