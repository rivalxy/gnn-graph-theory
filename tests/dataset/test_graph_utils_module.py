from pathlib import Path

import networkx as nx
import pytest
from sympy.combinatorics import Permutation, PermutationGroup

from dataset.graph_utils import (
    adj_to_nx_graph,
    bfs_expand_pseudo_similar,
    build_adjacency_dict,
    build_orbit_map,
    check_deletion_isomorphism,
    find_pseudo_similar_pair,
    is_extensible,
    is_injective,
    is_paut,
    paut_sizes_to_csv,
    DatasetType,
    PautStats,
    read_graphs_from_g6,
)


def test_read_graphs_from_g6(tmp_path: Path) -> None:
    g1 = nx.Graph()
    g1.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 4)])

    g2 = nx.Graph()
    g2.add_edges_from([(0, 1), (1, 2)])

    g6_path = tmp_path / "test_graphs.g6"
    with open(g6_path, "wb") as f:
        f.write(nx.to_graph6_bytes(g1, header=False))
        f.write(nx.to_graph6_bytes(g2, header=False))

    graphs = read_graphs_from_g6(str(g6_path))
    assert len(graphs) == 2
    assert graphs[0].num_of_nodes == 5
    assert graphs[0].adjacency_dict == {0: {1, 2}, 1: {0, 3}, 2: {0, 4}, 3: {1}, 4: {2}}
    assert graphs[1].num_of_nodes == 3


@pytest.mark.parametrize(
    "mapping, expected",
    [
        ({0: 0, 1: 1, 2: 2}, True),
        ({0: 0, 1: 1, 4: 4}, True),
        ({0: 2, 1: 1, 2: 0}, True),
        ({0: 0, 1: 1, 2: 2, 3: 4, 4: 3}, True),
        ({0: 3, 1: 1, 2: 4}, False),
        ({0: 1, 5: 6}, True),
        ({0: 1, 1: 0, 5: 6, 6: 5}, True),
        ({0: 5, 1: 1, 2: 2, 3: 3, 4: 4, 5: 0}, False),
        (dict(), False),
    ],
)
def test_is_paut(mapping: dict[int, int], expected: bool) -> None:
    adjacency_dict = {
        0: {1},
        1: {0, 2},
        2: {1, 3, 4},
        3: {2, 4},
        4: {2, 3, 5},
        5: {4, 6},
        6: {5},
    }
    assert is_paut(adjacency_dict, mapping) == expected


@pytest.fixture(scope="module")
def path_graph_group() -> PermutationGroup:
    from pynauty import Graph, autgrp

    adjacency_dict = {
        0: {1},
        1: {0, 2},
        2: {1, 3, 4},
        3: {2, 4},
        4: {2, 3, 5},
        5: {4, 6},
        6: {5},
    }
    graph = Graph(7)
    graph.set_adjacency_dict(adjacency_dict)
    generators_raw = autgrp(graph)[0]
    generators = [Permutation(g) for g in generators_raw]
    return PermutationGroup(generators)


@pytest.mark.parametrize(
    "mapping, expected",
    [
        ({0: 0, 1: 1, 2: 2}, True),
        ({0: 0, 1: 1, 4: 4}, True),
        ({0: 2, 1: 1, 2: 0}, False),
        ({0: 0, 1: 1, 2: 2, 3: 4, 4: 3}, False),
        ({}, True),
        ({0: 1, 5: 6}, False),
        ({0: 1, 1: 0, 5: 6, 6: 5}, False),
    ],
)
def test_is_extensible(
    path_graph_group: PermutationGroup, mapping: dict[int, int], expected: bool
) -> None:
    assert is_extensible(path_graph_group, mapping) == expected


# --- is_injective ---


@pytest.mark.parametrize(
    "mapping, expected",
    [
        ({0: 1, 1: 2, 2: 3}, True),
        ({0: 1, 1: 1}, False),
        ({5: 5}, True),
        ({}, True),
    ],
)
def test_is_injective(mapping: dict[int, int], expected: bool) -> None:
    assert is_injective(mapping) == expected


# --- build_adjacency_dict ---


def test_build_adjacency_dict_from_edges() -> None:
    adj = build_adjacency_dict([(0, 1), (1, 2)])
    assert adj == {0: {1}, 1: {0, 2}, 2: {1}}


def test_build_adjacency_dict_empty() -> None:
    assert build_adjacency_dict([]) == {}


# --- build_orbit_map ---


def test_build_orbit_map(path_graph_group: PermutationGroup) -> None:
    orbit_of = build_orbit_map(path_graph_group)
    # Automorphism is (0 6)(1 5)(2 4), so:
    assert orbit_of[0] == orbit_of[6]  # swapped
    assert orbit_of[1] == orbit_of[5]  # swapped
    assert orbit_of[2] == orbit_of[4]  # swapped
    assert orbit_of[3] != orbit_of[0]  # 3 is a fixed point, different orbit


# --- adj_to_nx_graph ---


def test_adj_to_nx_graph() -> None:
    adj = {0: {1}, 1: {0, 2}, 2: {1}}
    graph = adj_to_nx_graph(adj, num_nodes=3)
    assert graph.number_of_nodes() == 3
    assert set(graph.edges()) == {(0, 1), (1, 2)}


def test_adj_to_nx_graph_includes_isolated_nodes() -> None:
    graph = adj_to_nx_graph({}, num_nodes=3)
    assert graph.number_of_nodes() == 3
    assert graph.number_of_edges() == 0


# --- check_deletion_isomorphism ---


def test_check_deletion_isomorphism_symmetric_vertices() -> None:
    # Path 0-1-2: deleting 0 or 2 gives isomorphic graphs (both paths of length 1)
    graph = nx.path_graph(3)
    sigma = check_deletion_isomorphism(graph, 0, 2)
    assert sigma is not None
    # sigma maps V(G-0)={1,2} -> V(G-2)={0,1}
    assert 0 not in sigma
    assert 2 not in sigma.values()


def test_check_deletion_isomorphism_non_isomorphic() -> None:
    # Star graph: center=0, leaves=1,2,3. G-0 is 3 isolated nodes, G-1 is a star with 2 leaves.
    graph = nx.star_graph(3)
    assert check_deletion_isomorphism(graph, 0, 1) is None


# --- find_pseudo_similar_pair ---


def test_find_pseudo_similar_pair_returns_none_for_vertex_transitive() -> None:
    from pynauty import Graph, autgrp

    # Complete graph K4 is vertex-transitive: all nodes in one orbit, no cross-orbit pairs.
    adj = {0: {1, 2, 3}, 1: {0, 2, 3}, 2: {0, 1, 3}, 3: {0, 1, 2}}
    g = Graph(4)
    g.set_adjacency_dict(adj)
    generators = [Permutation(p) for p in autgrp(g)[0]]
    group = PermutationGroup(generators)

    assert find_pseudo_similar_pair(adj, group, 4) is None


# --- bfs_expand_pseudo_similar ---


def test_bfs_expand_pseudo_similar_contains_seed() -> None:
    adj = {0: {1}, 1: {0, 2}, 2: {1, 3}, 3: {2}}
    # sigma is an isomorphism V(G-0) -> V(G-3): the reversed path {1,2,3} -> {0,1,2}
    sigma = {1: 2, 2: 1, 3: 0}
    mapping = bfs_expand_pseudo_similar(adj, u=0, v=3, sigma=sigma, target_size=3)

    # Must contain the seed pair
    assert mapping[0] == 3
    # Must be a partial automorphism
    assert is_paut(adj, mapping)
    assert len(mapping) >= 1


# --- paut_sizes_to_csv ---


def test_paut_sizes_to_csv(tmp_path: Path) -> None:
    stats = {
        5: [PautStats(paut_size=3, label=1, dataset_type=DatasetType.TRAIN)],
        7: [PautStats(paut_size=4, label=0, dataset_type=DatasetType.VAL)],
    }
    csv_path = tmp_path / "stats.csv"
    paut_sizes_to_csv(stats, str(csv_path))

    lines = csv_path.read_text().strip().splitlines()
    assert lines[0] == "num_of_nodes,paut_size,label,dataset_type"
    assert "5,3,1,train" in lines[1]
    assert "7,4,0,val" in lines[2]
