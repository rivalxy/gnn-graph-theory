import csv
import random
from collections import deque
from dataclasses import dataclass
from enum import StrEnum
from typing import TypeAlias, cast

import networkx as nx
import pynauty
from sympy.combinatorics import Permutation, PermutationGroup

Edge: TypeAlias = tuple[int, int]
Mapping: TypeAlias = dict[int, int]
AdjacencyDict: TypeAlias = dict[int, set[int]]
OrbitMap: TypeAlias = dict[int, int]
Isomorphism: TypeAlias = dict[int, int]


def build_orbit_map(group: PermutationGroup) -> OrbitMap:
    return {node: i for i, orbit in enumerate(group.orbits()) for node in orbit}


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
    graph_data_list: list[GraphData] = []
    for graph in graphs:
        num_of_nodes = graph.number_of_nodes()
        adjacency_dict = build_adjacency_dict(list(graph.edges()))
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
    if not mapping:
        return True

    # Quick rejection: every src -> dst must have src and dst in the same orbit.
    orbit_of = build_orbit_map(group)
    for src, dst in mapping.items():
        if orbit_of.get(src) != orbit_of.get(dst):
            return False

    # Full check: enumerate group elements.
    domain = set(mapping.keys())
    for perm in group.generate():
        if all(cast(Permutation, perm).array_form[i] == mapping[i] for i in domain):
            return True
    return False


def adj_to_nx_graph(adj: AdjacencyDict, num_nodes: int) -> nx.Graph:
    """Convert an adjacency dictionary to a NetworkX graph.

    :param adj: Adjacency dictionary of the graph.
    :param num_nodes: Number of nodes.
    :returns: Equivalent NetworkX Graph.
    """
    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    for u, neighbors in adj.items():
        for v in neighbors:
            graph.add_edge(u, v)
    return graph


def check_deletion_isomorphism(graph: nx.Graph, u: int, v: int) -> Isomorphism | None:
    """Check whether G - u is isomorphic to G - v.

    If so, return a witnessing isomorphism sigma: V(G-u) -> V(G-v).
    sigma is defined on every vertex except u, and maps to every vertex
    except v.

    :param graph: A NetworkX graph.
    :param u: First vertex to delete.
    :param v: Second vertex to delete.
    :returns: Isomorphism dict or None if G-u ≇ G-v.
    """
    graph_minus_u = graph.copy()
    graph_minus_u.remove_node(u)
    graph_minus_v = graph.copy()
    graph_minus_v.remove_node(v)

    gm = nx.algorithms.isomorphism.GraphMatcher(graph_minus_u, graph_minus_v)
    if gm.is_isomorphic():
        return next(gm.isomorphisms_iter())
    return None


def find_pseudo_similar_pair(
    adj: AdjacencyDict,
    group: PermutationGroup,
    num_nodes: int,
    max_pairs: int = 100,
) -> tuple[int, int, dict[int, int]] | None:
    """Search for a pseudo-similar pair in the graph.

    A pseudo-similar pair (u, v) satisfies:
      - u and v are in *different* orbits of Aut(G)  → {u→v} is non-extendable
      - G - u ≅ G - v                                → the seed looks locally valid

    :param adj: Adjacency dictionary of the graph.
    :param group: Automorphism group of the graph.
    :param num_nodes: Number of nodes.
    :param max_pairs: Maximum cross-orbit pairs to test before giving up.
    :returns: (u, v, sigma) where sigma: V(G-u) -> V(G-v), or None if not found.
    """
    graph = adj_to_nx_graph(adj, num_nodes)
    orbit_of = build_orbit_map(group)

    nodes = list(range(num_nodes))
    random.shuffle(nodes)
    checked = 0

    for u in nodes:
        for v in nodes:
            if u >= v:
                continue

            if orbit_of.get(u) == orbit_of.get(v):
                continue

            sigma = check_deletion_isomorphism(graph, u, v)
            if sigma is not None:
                return u, v, sigma

            checked += 1
            if checked >= max_pairs:
                return None

    return None


def bfs_expand_pseudo_similar(
    adj: AdjacencyDict,
    u: int,
    v: int,
    sigma: Isomorphism,
    target_size: int,
) -> Mapping:
    """Grow the seed mapping {u: v} outward by BFS.

    For each visited node w, sigma[w] is the candidate image. The node is
    added to the domain only when w → sigma[w] keeps the mapping a valid
    partial automorphism of the *original* graph.

    Because u and v are pseudo-similar (different orbits), any mapping that
    contains u → v is guaranteed non-extendable, regardless of its size.

    :param adj: Adjacency dictionary of the original graph.
    :param u: Seed source vertex.
    :param v: Seed target vertex (image of u).
    :param sigma: Witnessing isomorphism V(G-u) -> V(G-v).
    :param target_size: Stop once the mapping reaches this size.
    :returns: A partial automorphism containing u → v, of size ≤ target_size.
    """
    mapping: Mapping = {u: v}
    used_targets: set[int] = {v}

    queue: deque[int] = deque([u])
    visited: set[int] = {u}

    while queue and len(mapping) < target_size:
        current = queue.popleft()
        neighbors = list(adj.get(current, set()))
        random.shuffle(neighbors)

        for w in neighbors:
            if w in visited:
                continue
            visited.add(w)

            candidate = sigma.get(w)
            if candidate is None or candidate in used_targets:
                continue

            test_map = {**mapping, w: candidate}
            if is_paut(adj, test_map):
                mapping = test_map
                used_targets.add(candidate)
                queue.append(w)

            if len(mapping) >= target_size:
                break

    return mapping


class DatasetType(StrEnum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


@dataclass
class PautStats:
    paut_size: int
    label: int
    dataset_type: DatasetType


def paut_sizes_to_csv(
    stats_by_node_count: dict[int, list[PautStats]], file_path: str
) -> None:
    """Write PautStats grouped by node count to a CSV file.

    :param stats_by_node_count: Dictionary mapping number of nodes to a list of PautStats.
    :param file_path: Path to the output CSV file.
    """
    with open(file_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["num_of_nodes", "paut_size", "label", "dataset_type"])
        for num_of_nodes, stats in stats_by_node_count.items():
            for stat in stats:
                writer.writerow(
                    [
                        num_of_nodes,
                        stat.paut_size,
                        stat.label,
                        stat.dataset_type,
                    ]
                )
