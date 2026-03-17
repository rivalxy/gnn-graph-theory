import pytest
from sympy.combinatorics import Permutation, PermutationGroup

from dataset.graph_utils import is_extensible, is_paut


@pytest.mark.skip(
    reason="Requires updated utils.py with GraphData and read_graphs_from_g6"
)
def test_read_graphs_from_g6() -> None:
    from dataset.graph_utils import read_graphs_from_g6

    graphs = read_graphs_from_g6("dataset/test_graphs.g6")
    assert len(graphs) == 3
    assert graphs[0].num_of_nodes == 5
    assert graphs[0].adjacency_dict == {0: {1, 2}, 1: {0, 3}, 2: {0, 4}, 3: {1}, 4: {2}}


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
