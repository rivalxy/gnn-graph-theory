import torch

from dataset.features import (
    BASELINE_FEATURE_DIM,
    EXTRA_FEATURE_DIM,
    FEATURE_NODE_ID,
    FEATURE_SOURCE_ID,
    FEATURE_TARGET_ID,
    build_extra_feature_matrix,
    make_pyg_data,
    normalize,
)


def test_normalize_handles_empty_tensor() -> None:
    values = torch.tensor([], dtype=torch.float)
    result = normalize(values)
    assert result.numel() == 0


def test_normalize_keeps_non_positive_values() -> None:
    values = torch.tensor([0.0, -1.0, -3.0], dtype=torch.float)
    result = normalize(values)
    assert torch.equal(result, values)


def test_normalize_scales_by_max() -> None:
    values = torch.tensor([1.0, 2.0, 4.0], dtype=torch.float)
    result = normalize(values)
    assert torch.allclose(result, torch.tensor([0.25, 0.5, 1.0]))


def test_build_extra_feature_matrix_shape_and_finite_values() -> None:
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]], dtype=torch.long
    )
    x = build_extra_feature_matrix(edge_index, num_of_nodes=3)
    assert x.shape == (3, EXTRA_FEATURE_DIM)
    assert torch.isfinite(x).all()


def test_make_pyg_data_baseline_features_encoding() -> None:
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    mapping = {0: 1}
    data = make_pyg_data(
        edge_index,
        num_of_nodes=2,
        mapping=mapping,
        label=1,
        extra_features=False,
    )

    assert data.x is not None
    assert data.x.shape == (2, BASELINE_FEATURE_DIM)
    assert data.y == 1.0
    assert data.x[0, FEATURE_NODE_ID].item() == 0.0
    assert data.x[1, FEATURE_NODE_ID].item() == 0.5
    assert data.x[0, FEATURE_TARGET_ID].item() == 0.5
    assert data.x[1, FEATURE_SOURCE_ID].item() == 0.0


def test_make_pyg_data_extra_features_shape() -> None:
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    data = make_pyg_data(
        edge_index,
        num_of_nodes=2,
        mapping={0: 1},
        label=0,
        extra_features=True,
    )

    assert data.x is not None
    assert data.x.shape == (2, EXTRA_FEATURE_DIM)
    assert data.y == 0.0
