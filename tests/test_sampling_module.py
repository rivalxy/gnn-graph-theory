import random

from pynauty import Graph, autgrp
from sympy.combinatorics import Permutation, PermutationGroup

from dataset.graph_utils import is_extensible, is_paut
from dataset.sampling import (
    block_automorphism,
    gen_blocking_examples,
    gen_positive_examples,
    gen_pseudo_similar_examples,
    is_identity_permutation,
    is_non_extensible_paut,
    mapping_key,
    partial_size_bounds,
    sample_partial_size,
)


def _group_from_adjacency(adjacency: dict[int, set[int]], n: int) -> PermutationGroup:
    graph = Graph(n)
    graph.set_adjacency_dict(adjacency)
    generators_raw = autgrp(graph)[0]
    generators = [Permutation(g) for g in generators_raw]
    return PermutationGroup(generators)


def test_is_identity_permutation() -> None:
    assert is_identity_permutation([0, 1, 2])
    assert not is_identity_permutation([1, 0, 2])


def test_mapping_key_is_stable() -> None:
    mapping = {1: 3, 2: 4}
    assert mapping_key(mapping) == mapping_key({2: 4, 1: 3})


def test_partial_size_bounds_and_sampling() -> None:
    min_size, max_size = partial_size_bounds(10)
    assert (min_size, max_size) == (5, 8)

    random.seed(42)
    for _ in range(20):
        value = sample_partial_size(10)
        assert 5 <= value <= 8


def test_gen_positive_examples_produce_extensible_pauts() -> None:
    random.seed(7)
    adjacency = {0: {1}, 1: {0, 2}, 2: {1, 3}, 3: {2}}
    group = _group_from_adjacency(adjacency, n=4)

    positives = gen_positive_examples(group, num_of_nodes=4, examples_num=4)

    assert len(positives) > 0
    for mapping, size in positives:
        assert len(mapping) == size
        assert is_paut(adjacency, mapping)
        assert is_extensible(group, mapping)


def test_gen_pseudo_similar_examples_returns_empty_on_tiny_graph() -> None:
    random.seed(11)
    # K3 + two attached vertices built via Godsil-Kocay construction.
    # With only 5 nodes the BFS can't grow mappings to the min size threshold,
    # so the function should return an empty list.
    adjacency: dict[int, set[int]] = {
        0: {1, 2},
        1: {0, 2},
        2: {0, 1},
        3: {0},
        4: {1},
    }
    group = _group_from_adjacency(adjacency, n=5)
    witness = {0: 1, 1: 0, 2: 2, 3: 4}  # maps G-v -> G-u

    negatives = gen_pseudo_similar_examples(
        group,
        num_of_nodes=5,
        adjacency_list=adjacency,
        u=3,
        v=4,
        witness=witness,
        examples_num=4,
    )

    assert negatives == []


def test_gen_blocking_examples_invariants() -> None:
    random.seed(19)
    adjacency = {0: {1}, 1: {0, 2}, 2: {1, 3}, 3: {2}}
    group = _group_from_adjacency(adjacency, n=4)

    negatives = gen_blocking_examples(
        group,
        examples_num=4,
        num_of_nodes=4,
        adjacency_list=adjacency,
    )

    for mapping, size in negatives:
        assert len(mapping) == size
        assert is_paut(adjacency, mapping)
        assert not is_extensible(group, mapping)


# --- is_non_extensible_paut ---


def test_is_non_extensible_paut_positive_case() -> None:
    # 7-node graph with automorphism (0 6)(1 5)(2 4).
    # {3: 4, 4: 3, 0: 6} is a valid paut but cannot extend to a full automorphism
    # because swapping 3<->4 while mapping 0->6 is inconsistent with (0 6)(1 5)(2 4).
    adjacency = {
        0: {1},
        1: {0, 2},
        2: {1, 3, 4},
        3: {2, 4},
        4: {2, 3, 5},
        5: {4, 6},
        6: {5},
    }
    group = _group_from_adjacency(adjacency, n=7)
    assert is_non_extensible_paut(adjacency, group, {3: 4, 4: 3, 0: 6})


def test_is_non_extensible_paut_extensible_mapping() -> None:
    # Path 0-1-2-3: the only non-trivial automorphism is (0 3)(1 2).
    # {0: 3, 1: 2} is a subset of it -> extensible -> not a non-extensible paut.
    adjacency = {0: {1}, 1: {0, 2}, 2: {1, 3}, 3: {2}}
    group = _group_from_adjacency(adjacency, n=4)
    assert not is_non_extensible_paut(adjacency, group, {0: 3, 1: 2})


def test_is_non_extensible_paut_invalid_mapping() -> None:
    # {0: 1} is not a paut on path 0-1-2-3 (0 and 1 are adjacent, but 1 and 1 trivially yes;
    # actually for a single pair it is vacuously a paut). Let's use a non-paut instead.
    adjacency = {0: {1}, 1: {0, 2}, 2: {1, 3}, 3: {2}}
    group = _group_from_adjacency(adjacency, n=4)
    # {0: 2}: 0-1 is an edge, but 2-1 is also an edge. 0-2 is not an edge, 2-? not relevant.
    # Actually need to pick a clear non-paut. {0: 1, 3: 1} is not injective.
    assert not is_non_extensible_paut(adjacency, group, {0: 1, 3: 1})


# --- block_automorphism ---


def test_block_automorphism_produces_non_extensible_result() -> None:
    random.seed(42)
    # 7-node graph with automorphism (0 6)(1 5)(2 4) — enough unmapped nodes
    # for blocking to find a cross-orbit assignment.
    adjacency = {
        0: {1},
        1: {0, 2},
        2: {1, 3, 4},
        3: {2, 4},
        4: {2, 3, 5},
        5: {4, 6},
        6: {5},
    }
    group = _group_from_adjacency(adjacency, n=7)

    positive = {0: 6, 1: 5}
    result = block_automorphism(positive, 7, adjacency, group)

    assert result is not None
    assert is_paut(adjacency, result)
    assert not is_extensible(group, result)
    for k, v in positive.items():
        assert result[k] == v


def test_block_automorphism_returns_none_when_fully_mapped() -> None:
    adjacency = {0: {1}, 1: {0}}
    group = _group_from_adjacency(adjacency, n=2)

    # All nodes already mapped — nothing left to block with
    assert block_automorphism({0: 1, 1: 0}, 2, adjacency, group) is None


def test_partial_size_bounds_with_upper_bound() -> None:
    min_size, max_size = partial_size_bounds(10, upper_bound=6)
    assert min_size == 5
    assert max_size == 6


def test_gen_pseudo_similar_examples_produces_non_extensible_pauts() -> None:
    random.seed(42)
    # Godsil-Kocay construction from C12 that produces verified pseudo-similar
    # vertices 12 and 13 (different orbits in the automorphism group of G).
    adj_G: dict[int, set[int]] = {
        0: {1, 11, 13},
        1: {0, 2},
        2: {1, 3, 12},
        3: {2, 4, 13},
        4: {3, 5, 12},
        5: {4, 6, 12, 13},
        6: {5, 7, 13},
        7: {6, 8},
        8: {7, 9, 12},
        9: {8, 10, 13},
        10: {9, 11, 12},
        11: {0, 10, 12, 13},
        12: {2, 4, 5, 8, 10, 11},
        13: {0, 3, 5, 6, 9, 11},
    }
    witness = {
        0: 7,
        1: 8,
        2: 9,
        3: 10,
        4: 11,
        5: 0,
        6: 1,
        7: 2,
        8: 3,
        9: 4,
        10: 5,
        11: 6,
        12: 13,
    }
    u, v = 12, 13
    group_G = _group_from_adjacency(adj_G, n=14)

    negatives = gen_pseudo_similar_examples(
        group_G,
        num_of_nodes=14,
        adjacency_list=adj_G,
        u=u,
        v=v,
        witness=witness,
        examples_num=5,
    )

    assert len(negatives) > 0
    for mapping, size in negatives:
        assert len(mapping) == size
        assert is_paut(adj_G, mapping)
        assert not is_extensible(group_G, mapping)
        assert mapping[u] == v
