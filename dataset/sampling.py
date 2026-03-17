import math
import random
from typing import cast

from sympy.combinatorics import Permutation, PermutationGroup

from dataset.graph_utils import (
    AdjacencyDict,
    Mapping,
    bfs_expand_pseudo_similar,
    build_orbit_map,
    find_pseudo_similar_pair,
    is_extensible,
    is_paut,
)

MAX_ATTEMPTS = 100
MIN_PARTIAL_AUT_FRACTION = 0.5
MAX_PARTIAL_AUT_FRACTION = 0.8
MAX_BLOCKING_CANDIDATES = 5


def is_identity_permutation(perm: list[int]) -> bool:
    return all(i == mapped for i, mapped in enumerate(perm))


def mapping_key(mapping: Mapping) -> frozenset[tuple[int, int]]:
    return frozenset(mapping.items())


def partial_size_bounds(
    num_of_nodes: int, upper_bound: int | None = None
) -> tuple[int, int]:
    min_size = math.ceil(num_of_nodes * MIN_PARTIAL_AUT_FRACTION)
    max_size = (
        upper_bound
        if upper_bound is not None
        else math.floor(num_of_nodes * MAX_PARTIAL_AUT_FRACTION)
    )
    return min_size, max_size


def sample_partial_size(num_of_nodes: int, upper_bound: int | None = None) -> int:
    min_size, max_size = partial_size_bounds(num_of_nodes, upper_bound)
    return random.randint(min_size, max_size)


def is_non_extensible_paut(
    adjacency_list: AdjacencyDict,
    group: PermutationGroup,
    mapping: Mapping,
) -> bool:
    return is_paut(adjacency_list, mapping) and not is_extensible(group, mapping)


def gen_positive_examples(
    group: PermutationGroup, num_of_nodes: int, examples_num: int
) -> list[tuple[Mapping, int]]:
    seen_positives: set[frozenset[tuple[int, int]]] = set()
    positives: list[tuple[Mapping, int]] = []
    attempts = 0
    nodes = list(range(num_of_nodes))

    while len(positives) < examples_num and attempts < MAX_ATTEMPTS * examples_num:
        attempts += 1
        perm = cast(Permutation, group.random()).array_form
        if is_identity_permutation(perm):
            continue

        p_aut_size = sample_partial_size(num_of_nodes)
        domain = random.sample(nodes, p_aut_size)
        mapping = {i: perm[i] for i in domain}

        key = mapping_key(mapping)
        if key in seen_positives:
            continue

        seen_positives.add(key)
        positives.append((mapping, len(mapping)))
    return positives


def gen_pseudo_similar_examples(
    group: PermutationGroup,
    num_of_nodes: int,
    adjacency_list: AdjacencyDict,
    examples_num: int,
) -> list[tuple[Mapping, int]]:
    """Generate hard negative examples using pseudo-similar vertex pairs."""
    pair = find_pseudo_similar_pair(adjacency_list, group, num_of_nodes)
    if pair is None:
        return []

    u, v, sigma = pair
    negatives: list[tuple[Mapping, int]] = []
    seen: set[frozenset[tuple[int, int]]] = set()

    min_size, max_size = partial_size_bounds(num_of_nodes)
    attempts = 0

    while len(negatives) < examples_num and attempts < MAX_ATTEMPTS * examples_num:
        attempts += 1
        target_size = random.randint(min_size, max_size)

        mapping = bfs_expand_pseudo_similar(adjacency_list, u, v, sigma, target_size)

        if len(mapping) < min_size:
            continue
        if not is_non_extensible_paut(adjacency_list, group, mapping):
            continue

        key = mapping_key(mapping)
        if key in seen:
            continue

        seen.add(key)
        negatives.append((mapping, len(mapping)))

    return negatives


def block_automorphism(
    positive: Mapping, num_of_nodes: int, adj: AdjacencyDict, group: PermutationGroup
) -> Mapping | None:
    nodes = list(range(num_of_nodes))
    orbit_of = build_orbit_map(group)

    unmapped_nodes = [n for n in nodes if n not in positive]
    targets = [n for n in nodes if n not in positive.values()]

    if not unmapped_nodes or not targets:
        return None

    random.shuffle(unmapped_nodes)
    random.shuffle(targets)

    for node in unmapped_nodes[: min(MAX_BLOCKING_CANDIDATES, len(unmapped_nodes))]:
        non_orbit_targets = [
            target for target in targets if orbit_of.get(target) != orbit_of.get(node)
        ]

        for target in non_orbit_targets[
            : min(MAX_BLOCKING_CANDIDATES, len(non_orbit_targets))
        ]:
            test_map = positive.copy()
            test_map[node] = target

            if is_non_extensible_paut(adj, group, test_map):
                return test_map

    return None


def gen_blocking_examples(
    group: PermutationGroup,
    examples_num: int,
    num_of_nodes: int,
    adjacency_list: AdjacencyDict,
) -> list[tuple[Mapping, int]]:
    """Generate non-extensible partial automorphisms using blocking strategy."""
    negatives: list[tuple[Mapping, int]] = []
    seen_negatives: set[frozenset[tuple[int, int]]] = set()
    attempts = 0
    nodes = list(range(num_of_nodes))
    maximum_size = math.floor(num_of_nodes * MAX_PARTIAL_AUT_FRACTION)

    while len(negatives) < examples_num and attempts < MAX_ATTEMPTS * examples_num:
        attempts += 1
        perm = cast(Permutation, group.random()).array_form
        if is_identity_permutation(perm):
            continue

        p_aut_size = sample_partial_size(num_of_nodes, upper_bound=maximum_size) - 1
        domain = random.sample(nodes, p_aut_size)
        mapping = {i: perm[i] for i in domain}

        maximum_extension = maximum_size - p_aut_size
        extension_size = random.randint(1, maximum_extension)
        current_extension = 0
        extension_attempts = 0

        while current_extension < extension_size and extension_attempts < MAX_ATTEMPTS:
            extension_attempts += 1
            new_mapping = block_automorphism(
                mapping, num_of_nodes, adjacency_list, group
            )
            if new_mapping is None:
                break

            mapping = new_mapping
            current_extension += 1

        if not is_non_extensible_paut(adjacency_list, group, mapping):
            continue

        key = mapping_key(mapping)
        if key in seen_negatives:
            continue

        seen_negatives.add(key)
        negatives.append((mapping, len(mapping)))

    return negatives
