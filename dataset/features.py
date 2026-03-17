import networkx as nx
import torch
from data_utils import Mapping
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

BASELINE_FEATURE_DIM = 3
EXTRA_FEATURE_DIM = 7

FEATURE_NODE_ID = 0
FEATURE_TARGET_ID = 1
FEATURE_SOURCE_ID = 2
FEATURE_DEGREE = 3
FEATURE_CLUSTERING = 4
FEATURE_TRIANGLES = 5
FEATURE_AVG_NEIGHBOR_DEGREE = 6


def normalize(values: torch.Tensor) -> torch.Tensor:
    if values.numel() == 0:
        return values

    max_value = values.max().item()
    if max_value <= 0:
        return values

    return values / max_value


def build_extra_feature_matrix(
    tensor_edge_index: torch.Tensor, num_of_nodes: int
) -> torch.Tensor:
    x = torch.full((num_of_nodes, EXTRA_FEATURE_DIM), -1.0, dtype=torch.float)
    pyg_graph = Data(edge_index=tensor_edge_index, num_nodes=num_of_nodes)
    nx_graph = to_networkx(pyg_graph, to_undirected=True)

    degrees = torch.tensor(
        [nx_graph.degree(node) for node in range(num_of_nodes)], dtype=torch.float
    )
    x[:, FEATURE_DEGREE] = normalize(degrees)

    clustering_coeffs = torch.tensor(
        [nx.clustering(nx_graph, node) for node in range(num_of_nodes)],
        dtype=torch.float,
    )
    x[:, FEATURE_CLUSTERING] = clustering_coeffs

    triangle_counts = torch.tensor(
        [nx.triangles(nx_graph, node) for node in range(num_of_nodes)],
        dtype=torch.float,
    )
    x[:, FEATURE_TRIANGLES] = normalize(triangle_counts)

    avg_neighbor_degrees = []
    for node in range(num_of_nodes):
        neighbors = list(nx_graph.neighbors(node))
        if neighbors:
            avg_degree = sum(nx_graph.degree(neighbor) for neighbor in neighbors) / len(
                neighbors
            )
        else:
            avg_degree = 0.0
        avg_neighbor_degrees.append(avg_degree)

    x[:, FEATURE_AVG_NEIGHBOR_DEGREE] = normalize(
        torch.tensor(avg_neighbor_degrees, dtype=torch.float)
    )
    return x


def make_pyg_data(
    tensor_edge_index: torch.Tensor,
    num_of_nodes: int,
    mapping: Mapping,
    label: int,
    extra_features: bool,
) -> Data:
    """Create a PyTorch Geometric Data object from a partial automorphism mapping."""
    if extra_features:
        x = build_extra_feature_matrix(tensor_edge_index, num_of_nodes)
    else:
        x = torch.full((num_of_nodes, BASELINE_FEATURE_DIM), -1.0, dtype=torch.float)

    for node in range(num_of_nodes):
        x[node, FEATURE_NODE_ID] = float(node) / num_of_nodes

    for source, target in mapping.items():
        x[source, FEATURE_TARGET_ID] = float(target) / num_of_nodes
        x[target, FEATURE_SOURCE_ID] = float(source) / num_of_nodes

    return Data(
        x=x, edge_index=tensor_edge_index, y=torch.tensor([label], dtype=torch.float)
    )
