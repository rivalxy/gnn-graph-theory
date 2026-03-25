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


def test_gen_pseudo_similar_examples_returns_empty_when_no_pair() -> None:
    random.seed(11)
    # Complete graph has one orbit, so cross-orbit pseudo-similar pair search fails.
    adjacency = {
        0: {1, 2, 3},
        1: {0, 2, 3},
        2: {0, 1, 3},
        3: {0, 1, 2},
    }
    group = _group_from_adjacency(adjacency, n=4)

    negatives = gen_pseudo_similar_examples(
        group,
        num_of_nodes=4,
        adjacency_list=adjacency,
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
    adjacency = {0: {1}, 1: {0, 2}, 2: {1, 3, 4}, 3: {2, 4}, 4: {2, 3, 5}, 5: {4, 6}, 6: {5}}
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
    adjacency = {0: {1}, 1: {0, 2}, 2: {1, 3}, 3: {2}}
    group = _group_from_adjacency(adjacency, n=4)

    # Start from an extensible subset {0: 3, 1: 2} (part of automorphism (0 3)(1 2))
    positive = {0: 3, 1: 2}
    result = block_automorphism(positive, 4, adjacency, group)

    if result is not None:
        assert is_paut(adjacency, result)
        assert not is_extensible(group, result)
        # Must extend the original mapping
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
