import math
import random

import networkx as nx
from utils import AdjacencyDict, Edge, GraphData, Mapping, build_adjacency_dict, is_paut

MAX_ATTEMPTS = 100
MIN_PARTIAL_AUT_FRACTION = 0.5
MAX_PARTIAL_AUT_FRACTION = 0.8
MAX_PSEUDO_GRAPHS = 3
EDGE_SWAP_MULTIPLIER = 50


def sample_partial_size(num_of_nodes: int) -> int:
    min_size = math.ceil(num_of_nodes * MIN_PARTIAL_AUT_FRACTION)
    max_size = math.floor(num_of_nodes * MAX_PARTIAL_AUT_FRACTION)
    return random.randint(min_size, max_size)


def generate_pseudosimilar_graph(
    edge_list: list[Edge], num_of_nodes: int, num_swaps: int = 3
) -> tuple[list[Edge] | None, bool]:
    """Generate a pseudosimilar (non-isomorphic) graph via edge swaps.

    Maintains degree sequence but changes graph structure.

    :param edge_list: List of edges in the original graph.
    :param num_of_nodes: Number of nodes in the graph.
    :param num_swaps: Target number of successful edge swaps.
    :returns: Tuple of (new_edge_list, success) where success indicates non-isomorphism.
    """
    # Convert to NetworkX for easier manipulation
    G = nx.Graph()
    G.add_nodes_from(range(num_of_nodes))
    G.add_edges_from(edge_list)

    G_modified = G.copy()
    edges = list(G_modified.edges())

    if len(edges) < 2:
        return None, False

    successful_swaps = 0
    attempts = 0
    max_attempts = num_swaps * EDGE_SWAP_MULTIPLIER

    while successful_swaps < num_swaps and attempts < max_attempts:
        attempts += 1

        # Pick two random edges
        if len(edges) < 2:
            break

        (a, b), (c, d) = random.sample(edges, 2)

        # Ensure all nodes are distinct
        if len({a, b, c, d}) < 4:
            continue

        # Try swap: (a,b), (c,d) -> (a,c), (b,d)
        if not G_modified.has_edge(a, c) and not G_modified.has_edge(b, d):
            G_modified.remove_edge(a, b)
            G_modified.remove_edge(c, d)
            G_modified.add_edge(a, c)
            G_modified.add_edge(b, d)

            successful_swaps += 1
            edges = list(G_modified.edges())

    # Verify non-isomorphism
    if successful_swaps > 0 and not nx.is_isomorphic(G, G_modified):
        new_edge_list = list(G_modified.edges())
        return new_edge_list, True

    return None, False


def create_cross_graph_mapping(
    edge_list1: list[Edge], edge_list2: list[Edge], num_of_nodes: int
) -> Mapping | None:
    """Create a partial mapping between two non-isomorphic graphs.

    This mapping is locally valid but cannot extend (graphs aren't isomorphic).

    :param edge_list1: Edge list of the first graph.
    :param edge_list2: Edge list of the second graph.
    :param num_of_nodes: Number of nodes in both graphs.
    :returns: A partial mapping if successful, None otherwise.
    """
    # Build adjacency for both graphs
    adj1 = build_adjacency_dict(edge_list1)
    adj2 = build_adjacency_dict(edge_list2)

    # Compute degree signatures for both graphs
    nodes = list(range(num_of_nodes))

    degree1 = {n: len(adj1.get(n, set())) for n in nodes}
    degree2 = {n: len(adj2.get(n, set())) for n in nodes}

    # Group nodes by degree in both graphs
    degree_groups1: dict[int, list[int]] = {}
    for node, deg in degree1.items():
        degree_groups1.setdefault(deg, []).append(node)

    degree_groups2: dict[int, list[int]] = {}
    for node, deg in degree2.items():
        degree_groups2.setdefault(deg, []).append(node)

    # Build mapping by matching nodes with same degree
    mapping: Mapping = {}
    map_size = sample_partial_size(num_of_nodes)
    used_targets = set()

    for degree in sorted(degree_groups1.keys()):
        if degree not in degree_groups2:
            continue

        group1 = degree_groups1[degree].copy()
        group2 = [n for n in degree_groups2[degree] if n not in used_targets]

        random.shuffle(group1)
        random.shuffle(group2)

        for src in group1:
            if len(mapping) >= map_size:
                break

            if src in mapping:
                continue

            # Find compatible target
            for tgt in group2:
                if tgt in used_targets:
                    continue

                # Check local validity
                test_map = mapping.copy()
                test_map[src] = tgt

                valid = True
                src_neighbors = adj1.get(src, set())
                tgt_neighbors = adj2.get(tgt, set())

                for neighbor in src_neighbors:
                    if neighbor in test_map:
                        if test_map[neighbor] not in tgt_neighbors:
                            valid = False
                            break

                if valid:
                    mapping[src] = tgt
                    used_targets.add(tgt)
                    break

        if len(mapping) >= map_size:
            break

    return mapping if len(mapping) >= 2 else None


def gen_negative_examples_with_pseudosimilar(
    num_of_nodes: int,
    edge_list: list[Edge],
    examples_num: int,
) -> list[tuple[str, Mapping, list[Edge]]]:
    """Generate negative examples using cross-graph mappings to pseudosimilar graphs.

    :param num_of_nodes: Number of nodes in the graph.
    :param edge_list: Edge list of the original graph.
    :param examples_num: Number of negative examples to generate.
    :returns: List of tuples ("pseudo", mapping, pseudo_edge_list).
    """
    negatives = []
    seen_negatives = set()

    attempts = 0
    generated_pseudo: list[list[Edge]] = []

    # Generate multiple pseudosimilar graphs
    for swap_count in [2, 3, 4, 5]:
        pseudo_edge_list, success = generate_pseudosimilar_graph(
            edge_list, num_of_nodes, num_swaps=swap_count
        )
        if success and pseudo_edge_list:
            generated_pseudo.append(pseudo_edge_list)

        # Generate up to 3 different pseudosimilar graphs
        if len(generated_pseudo) >= MAX_PSEUDO_GRAPHS:
            break

    adjacency_dict = build_adjacency_dict(edge_list)
    while len(negatives) < examples_num and attempts < MAX_ATTEMPTS * examples_num:
        attempts += 1

        if not generated_pseudo:
            break
        pseudo_edge_list = random.choice(generated_pseudo)
        mapping = create_cross_graph_mapping(edge_list, pseudo_edge_list, num_of_nodes)

        if mapping is None:
            continue

        # Verify local validity on original graph
        if not is_paut(adjacency_dict, mapping):
            continue

        key = frozenset(mapping.items())
        if key in seen_negatives:
            continue

        seen_negatives.add(key)
        negatives.append(("pseudo", mapping, pseudo_edge_list))

    return negatives


def generate_testset(
    graph_data_list: list[GraphData], examples_per_graph: int
) -> list[tuple[AdjacencyDict, list[tuple[str, Mapping, list[Edge]]]]]:
    """Generate a test set with negative examples from pseudosimilar graphs.

    :param graph_data_list: List of graph data objects.
    :param examples_per_graph: Number of negative examples per graph.
    :returns: List of tuples (adjacency_dict, negatives).
    """
    testset = []
    for graph_data in graph_data_list:
        edge_list = [
            (u, v)
            for u, neighbors in graph_data.adjacency_dict.items()
            for v in neighbors
            if u < v
        ]
        negatives = gen_negative_examples_with_pseudosimilar(
            graph_data.num_of_nodes, edge_list, examples_per_graph
        )
        testset.append((graph_data.num_of_nodes, graph_data.adjacency_dict, negatives))
    return testset
