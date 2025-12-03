import networkx as nx
from pynauty import Graph, autgrp
from sklearn.model_selection import train_test_split
import random
import torch
from torch_geometric.data import Data
from sympy.combinatorics import Permutation, PermutationGroup


MAX_EXAMPLES_NUM = 30
MAX_ATTEMPTS = 100


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


def generate_partial_automorphism_graphs(graphs: list[Graph]) -> list:
    """
    Generates partial automorphism graphs from a list of pynauty graphs.

    :param graphs: List of pynauty graphs.
    :returns: #TODO
    """

    dataset = []

    for Graph, num_of_nodes, edge_list in graphs:
        generators_raw, grpsize1, grpsize2, _, _ = autgrp(Graph)
        group_size = grpsize1 * 10**grpsize2
        
        # ensure 10-30 examples per graph with 1:1 ratio of positive to negative examples
        examples_num = int(min(MAX_EXAMPLES_NUM, group_size))
        generators = [Permutation(g) for g in generators_raw]
        group = PermutationGroup(generators)

        seen_positives = set()
        seen_negatives = set()

        # positive examples
        positives = []
        attempts = 0
        while len(positives) < examples_num and attempts < MAX_ATTEMPTS * examples_num:
            attempts += 1
            perm = group.random().array_form

            p_aut_size = random.randint(max(3, num_of_nodes // 3), max(4, 2 * num_of_nodes // 3))
            p_aut_size = min(p_aut_size, num_of_nodes)
            domain = random.sample(range(num_of_nodes), p_aut_size)
            mapping = {i: perm[i] for i in domain}

            key = frozenset(mapping.items())
            if key in seen_positives:
                continue
            seen_positives.add(key)
            positives.append(mapping)
            dataset.append(_make_data(edge_list, num_of_nodes, mapping, label=1))

        #FIXME negative examples
        for mapping in positives:
            u = random.choice(list(mapping.keys()))
            v_old = mapping[u]
            v_new = random.choice([v for v in range(num_of_nodes) if v != v_old])
            neg_map = mapping.copy()
            neg_map[u] = v_new
            key = frozenset(neg_map.items())
            if key in seen_negatives:
                continue
            seen_negatives.add(key)
            dataset.append(_make_data(edge_list, num_of_nodes, neg_map, label=0))

    return dataset


def _make_data(edge_list: list[tuple], num_of_nodes: int,  mapping: dict[int, int], label: int) -> Data:
    # 3 features: node_id, source_id (if mapped), target_id (if mapped)
    x = torch.full((num_of_nodes, 3), -1.0, dtype=torch.float)
    
    # Give EVERY node its own identity
    for node in range(num_of_nodes):
        x[node, 0] = float(node) / num_of_nodes  
    
    # Mark mapped nodes with bidirectional info
    for source, target in mapping.items():
        x[source, 1] = float(target) / num_of_nodes  # target_id
        x[target, 2] = float(source) / num_of_nodes  # source_id

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
    return data


if __name__ == "__main__":
    all_graphs = read_graphs_from_g6("dataset/2000_raw_graphs.g6")
    graphs_train, graphs_val = train_test_split(all_graphs, test_size=0.2)
    train_dataset = generate_partial_automorphism_graphs(graphs_train)
    val_dataset = generate_partial_automorphism_graphs(graphs_val)
    print(
        f"Generated {len(train_dataset)} train examples and {len(val_dataset)} val examples.")
