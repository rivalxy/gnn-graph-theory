import random

from pynauty import Graph, autgrp
from sympy.combinatorics import Permutation, PermutationGroup

from dataset.graph_utils import is_extensible, is_paut
from dataset.sampling import (
    gen_blocking_examples,
    gen_positive_examples,
    gen_pseudo_similar_examples,
    is_identity_permutation,
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
