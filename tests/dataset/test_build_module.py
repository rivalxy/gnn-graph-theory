import torch

from dataset.build import RawPautExample, build_edge_index, raw_examples_to_pyg
from dataset.graph_utils import DatasetType, PautStats


def test_build_edge_index_empty_graph() -> None:
    edge_index = build_edge_index({})
    assert edge_index.shape == (2, 0)


def test_raw_examples_to_pyg_converts_and_aggregates_stats() -> None:
    raw_examples = [
        RawPautExample(
            edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
            num_of_nodes=2,
            mapping={0: 1},
            label=1,
            paut_stats=PautStats(
                paut_size=1,
                label=1,
                dataset_type=DatasetType.TRAIN,
            ),
        )
    ]

    pyg_data, paut_sizes = raw_examples_to_pyg(raw_examples, extra_features=False)

    assert len(pyg_data) == 1
    assert pyg_data[0].num_nodes == 2
    assert pyg_data[0].y == 1.0
    assert 2 in paut_sizes
    assert len(paut_sizes[2]) == 1
    assert paut_sizes[2][0].paut_size == 1
