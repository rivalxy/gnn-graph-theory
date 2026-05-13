import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.data import Data

from utils import (
    FEATURE_SOURCE_ID,
    FEATURE_TARGET_ID,
    build_predictions_df,
    can_map_vertex,
    compute_avg_mapping_distance,
    compute_diameter,
    decode_mapping,
    error_by_aut_grp,
    has_degree_mismatch,
    is_vertex_transitive,
    load_json,
    paut_size_from_torch,
    regularity_check,
)

FEATURE_NODE_ID = 0


def make_graph(
    edge_list: list[tuple[int, int]],
    num_nodes: int,
    mapping: dict[int, int] | None = None,
) -> Data:
    """Build a synthetic Data object with a 3-column feature matrix."""
    if edge_list:
        fwd_src, fwd_dst = zip(*edge_list)
        src_all = list(fwd_src) + list(fwd_dst)
        dst_all = list(fwd_dst) + list(fwd_src)
        edge_index = torch.tensor([src_all, dst_all], dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    x = torch.full((num_nodes, 3), -1.0)
    for node in range(num_nodes):
        x[node, FEATURE_NODE_ID] = node / num_nodes

    if mapping:
        for src_node, tgt_node in mapping.items():
            x[src_node, FEATURE_TARGET_ID] = tgt_node / num_nodes
            x[tgt_node, FEATURE_SOURCE_ID] = src_node / num_nodes

    return Data(x=x, edge_index=edge_index, num_nodes=num_nodes)


# ── paut_size_from_torch ───────────────────────────────────────────────────────


def test_paut_size_no_feature_matrix() -> None:
    graph = Data(x=None, edge_index=torch.zeros((2, 0), dtype=torch.long), num_nodes=3)
    assert paut_size_from_torch(graph) == 0


def test_paut_size_empty_mapping() -> None:
    graph = make_graph([(0, 1), (1, 2)], num_nodes=3)
    assert paut_size_from_torch(graph) == 0


def test_paut_size_two_mapped_nodes() -> None:
    # Mapping {0: 2, 1: 3} on a 5-node path
    graph = make_graph(
        [(0, 1), (1, 2), (2, 3), (3, 4)],
        num_nodes=5,
        mapping={0: 2, 1: 3},
    )
    assert paut_size_from_torch(graph) == 2


# ── regularity_check ──────────────────────────────────────────────────────────


def test_regularity_none_edge_index_is_regular() -> None:
    graph = Data(edge_index=None, num_nodes=3)
    assert regularity_check(graph) is True


def test_regularity_triangle_is_regular() -> None:
    # K3: all nodes have degree 2
    graph = make_graph([(0, 1), (0, 2), (1, 2)], num_nodes=3)
    assert regularity_check(graph) is True


def test_regularity_star_is_not_regular() -> None:
    # K1,3: center has degree 3, leaves have degree 1
    graph = make_graph([(0, 1), (0, 2), (0, 3)], num_nodes=4)
    assert regularity_check(graph) is False


def test_regularity_no_edges_is_regular() -> None:
    # All degrees are 0
    graph = make_graph([], num_nodes=4)
    assert regularity_check(graph) is True


# ── decode_mapping ────────────────────────────────────────────────────────────


def test_decode_mapping_empty() -> None:
    graph = make_graph([(0, 1), (1, 2), (2, 3)], num_nodes=4)
    assert decode_mapping(graph) == {}


def test_decode_mapping_recovers_encoded_mapping() -> None:
    # Path 0-1-2-3-4 with mapping {0: 4, 1: 3}
    graph = make_graph(
        [(0, 1), (1, 2), (2, 3), (3, 4)],
        num_nodes=5,
        mapping={0: 4, 1: 3},
    )
    result = decode_mapping(graph)
    assert result == {0: 4, 1: 3}


def test_decode_mapping_self_loop() -> None:
    # Node 0 maps to itself
    graph = make_graph([(0, 1), (1, 2)], num_nodes=3, mapping={0: 0})
    result = decode_mapping(graph)
    assert result == {0: 0}


# ── build_predictions_df ──────────────────────────────────────────────────────


@pytest.fixture
def sample_records() -> list[dict]:
    return [
        {
            "sample_idx": 0,
            "num_nodes": 4,
            "regular": True,
            "paut_size": 2,
            "aut_grp_order": 8,
            "true_label": 1,
            "prediction": 1,
            "pred_prob": 0.9,
            "correct": True,
        },
        {
            "sample_idx": 1,
            "num_nodes": 6,
            "regular": False,
            "paut_size": 3,
            "aut_grp_order": 2,
            "true_label": 0,
            "prediction": 1,
            "pred_prob": 0.7,
            "correct": False,
        },
    ]


def test_build_predictions_df_has_derived_columns(sample_records) -> None:
    df = build_predictions_df(sample_records)
    assert "paut_relative_size" in df.columns
    assert "error" in df.columns


def test_build_predictions_df_paut_relative_size(sample_records) -> None:
    df = build_predictions_df(sample_records)
    assert df.loc[0, "paut_relative_size"] == pytest.approx(2 / 4)
    assert df.loc[1, "paut_relative_size"] == pytest.approx(3 / 6)


def test_build_predictions_df_error_flag(sample_records) -> None:
    df = build_predictions_df(sample_records)
    assert df.loc[0, "error"] == 0  # correct prediction
    assert df.loc[1, "error"] == 1  # wrong prediction


# ── error_by_aut_grp ──────────────────────────────────────────────────────────


def test_error_by_aut_grp_buckets_correctly() -> None:
    df = pd.DataFrame(
        {
            "aut_grp_order": [1, 2, 5, 10],
            "error": [0, 1, 0, 1],
        }
    )
    # Two bins: (0, 3] → values 1,2 and (3, 12] → values 5,10
    edges = [0, 3, 12]
    result = error_by_aut_grp(df, edges)

    assert "error_rate" in result.columns
    assert "count" in result.columns
    assert len(result) == 2
    assert (
        result.loc[result["bucket"].astype(str).str.contains("3.0, 12"), "count"].iloc[
            0
        ]
        == 2
    )


def test_error_by_aut_grp_does_not_mutate_input() -> None:
    df = pd.DataFrame({"aut_grp_order": [1, 2], "error": [0, 1]})
    original_cols = list(df.columns)
    error_by_aut_grp(df, [0, 2, 5])
    assert list(df.columns) == original_cols


# ── can_map_vertex ────────────────────────────────────────────────────────────


@pytest.fixture
def triangle() -> nx.Graph:
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (0, 2)])
    return G


@pytest.fixture
def star() -> nx.Graph:
    # Center 0 connected to leaves 1, 2, 3
    G = nx.Graph()
    G.add_edges_from([(0, 1), (0, 2), (0, 3)])
    return G


def test_can_map_vertex_triangle_any_to_any(triangle) -> None:
    for u in range(3):
        for v in range(3):
            assert can_map_vertex(triangle, u, v) is True


def test_can_map_vertex_star_center_to_leaf_false(star) -> None:
    # Degree mismatch: center has degree 3, leaf has degree 1
    assert can_map_vertex(star, 0, 1) is False


def test_can_map_vertex_star_leaf_to_leaf_true(star) -> None:
    # All leaves have degree 1 and the swap permutation is an automorphism
    assert can_map_vertex(star, 1, 2) is True
    assert can_map_vertex(star, 1, 3) is True


# ── is_vertex_transitive ──────────────────────────────────────────────────────


def test_is_vertex_transitive_triangle() -> None:
    G = nx.cycle_graph(3)
    assert is_vertex_transitive(G) is True


def test_is_vertex_transitive_cycle_c4() -> None:
    G = nx.cycle_graph(4)
    assert is_vertex_transitive(G) is True


def test_is_vertex_transitive_star_false() -> None:
    G = nx.star_graph(3)  # K1,3: center + 3 leaves, not regular
    assert is_vertex_transitive(G) is False


def test_is_vertex_transitive_path_p3_false() -> None:
    G = nx.path_graph(3)  # degrees 1, 2, 1 — not regular
    assert is_vertex_transitive(G) is False


# ── has_degree_mismatch ───────────────────────────────────────────────────────


def test_has_degree_mismatch_false_when_degrees_match() -> None:
    # C4 with mapping {0: 2}: both have degree 2
    graph = make_graph([(0, 1), (1, 2), (2, 3), (3, 0)], num_nodes=4, mapping={0: 2})
    assert has_degree_mismatch(graph) is False


def test_has_degree_mismatch_true_when_degrees_differ() -> None:
    # Star K1,3: center (degree 3) maps to leaf (degree 1)
    graph = make_graph([(0, 1), (0, 2), (0, 3)], num_nodes=4, mapping={0: 1})
    assert has_degree_mismatch(graph) is True


def test_has_degree_mismatch_false_for_empty_mapping() -> None:
    graph = make_graph([(0, 1), (1, 2)], num_nodes=3)
    assert has_degree_mismatch(graph) is False


# ── compute_avg_mapping_distance ──────────────────────────────────────────────


def test_compute_avg_mapping_distance_empty_mapping_returns_nan() -> None:
    graph = make_graph([(0, 1), (1, 2), (2, 3)], num_nodes=4)
    result = compute_avg_mapping_distance(graph)
    assert np.isnan(result)


def test_compute_avg_mapping_distance_adjacent_pair() -> None:
    # Path 0-1-2-3; mapping {0: 1}: distance 1
    graph = make_graph([(0, 1), (1, 2), (2, 3)], num_nodes=4, mapping={0: 1})
    assert compute_avg_mapping_distance(graph) == pytest.approx(1.0)


def test_compute_avg_mapping_distance_far_pair() -> None:
    # Path 0-1-2-3; mapping {0: 3}: distance 3
    graph = make_graph([(0, 1), (1, 2), (2, 3)], num_nodes=4, mapping={0: 3})
    assert compute_avg_mapping_distance(graph) == pytest.approx(3.0)


def test_compute_avg_mapping_distance_averages_multiple_pairs() -> None:
    # Path 0-1-2-3; mapping {0: 1, 2: 3}: distances 1 and 1, avg = 1.0
    graph = make_graph([(0, 1), (1, 2), (2, 3)], num_nodes=4, mapping={0: 1, 2: 3})
    assert compute_avg_mapping_distance(graph) == pytest.approx(1.0)


# ── compute_diameter ──────────────────────────────────────────────────────────


def test_compute_diameter_path() -> None:
    # Path 0-1-2-3: diameter 3
    graph = make_graph([(0, 1), (1, 2), (2, 3)], num_nodes=4)
    assert compute_diameter(graph) == 3


def test_compute_diameter_triangle() -> None:
    graph = make_graph([(0, 1), (1, 2), (0, 2)], num_nodes=3)
    assert compute_diameter(graph) == 1


def test_compute_diameter_disconnected_uses_largest_component() -> None:
    # Two disconnected edges: 0-1 and 2-3
    # Largest component has 2 nodes, diameter 1
    graph = make_graph([(0, 1), (2, 3)], num_nodes=4)
    assert compute_diameter(graph) == 1


# ── load_json ─────────────────────────────────────────────────────────────────


def test_load_json_returns_dict(tmp_path: Path) -> None:
    data = {"key": "value", "number": 42}
    json_path = tmp_path / "config.json"
    json_path.write_text(json.dumps(data))
    result = load_json(str(json_path))
    assert result == data


def test_load_json_nested(tmp_path: Path) -> None:
    data = {"model": {"hidden_dim": 64, "num_layers": 3}, "lr": 0.001}
    json_path = tmp_path / "nested.json"
    json_path.write_text(json.dumps(data))
    result = load_json(str(json_path))
    assert result["model"]["hidden_dim"] == 64
