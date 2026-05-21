import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pynauty
import seaborn as sns
import torch
import torch.nn as nn
import torch_geometric.data
import torch_geometric.loader
import torch_geometric.utils
from IPython.display import display
from scipy.stats import chi2_contingency, spearmanr
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

from dataset.features import FEATURE_SOURCE_ID, FEATURE_TARGET_ID
from dataset.graph_utils import build_adjacency_dict
from models import GIN, GPS

PE_DIM = 5
ATTN_HEADS = 4


def paut_size_from_torch(torch_graph: torch_geometric.data.Data) -> int:
    x = torch_graph.x
    if x is None:
        return 0
    paut_size = int((x[:, FEATURE_TARGET_ID] != -1).sum().item())
    assert (x[:, FEATURE_TARGET_ID] != -1).sum() == (
        x[:, FEATURE_SOURCE_ID] != -1
    ).sum()
    return paut_size


def aut_grp_order_from_torch(torch_graph: torch_geometric.data.Data) -> int:
    num_nodes = torch_graph.num_nodes
    nx_graph = torch_geometric.utils.to_networkx(torch_graph)
    pynauty_graph = pynauty.Graph(num_nodes, directed=False)
    adjacency_dict = build_adjacency_dict(nx_graph.edges())
    pynauty_graph.set_adjacency_dict(adjacency_dict)
    _, grpsize1, grpsize2, _, _ = pynauty.autgrp(pynauty_graph)
    return int(round(grpsize1 * 10**grpsize2))


def regularity_check(graph: torch_geometric.data.Data) -> bool:
    if graph.edge_index is None:
        return True
    degrees = torch_geometric.utils.degree(
        graph.edge_index[0], num_nodes=graph.num_nodes
    )
    return bool((degrees == degrees[0]).all().item())


def load_or_compute_dataset_metadata(
    dataset: list[torch_geometric.data.Data], dataset_path: str
) -> list[dict[str, Any]]:
    """Compute per-graph metadata (aut_grp_order, regularity, etc.) with file caching.

    The cache is stored next to the dataset file as ``<name>_metadata_cache.json``.
    """
    cache_path = Path(dataset_path).with_name(
        Path(dataset_path).stem + "_metadata_cache.json"
    )

    if cache_path.exists():
        with open(cache_path) as f:
            cached = json.load(f)
        if len(cached) == len(dataset):
            return cached

    metadata = []
    for graph in dataset:
        metadata.append(
            {
                "num_nodes": graph.num_nodes,
                "regular": regularity_check(graph),
                "paut_size": paut_size_from_torch(graph),
                "aut_grp_order": aut_grp_order_from_torch(graph),
            }
        )

    with open(cache_path, "w") as f:
        json.dump(metadata, f)

    return metadata


def evaluate_checkpoint(
    config_path: str, dataset_path: str, checkpoint_path: str
) -> dict[str, Any]:
    config = load_json(config_path)

    evaluation_dataset = torch.load(dataset_path, weights_only=False)

    dataset_metadata = load_or_compute_dataset_metadata(
        evaluation_dataset, dataset_path
    )

    is_gps = "num_heads" in config

    evaluation_loader = torch_geometric.loader.DataLoader(
        evaluation_dataset, batch_size=config["batch_size"], shuffle=False
    )

    number_of_features = evaluation_dataset[0].num_node_features
    if is_gps:
        evaluation_model: nn.Module = GPS(
            number_of_features,
            config["hidden_dim"],
            config["num_layers"],
            config["dropout"],
            ATTN_HEADS,
            PE_DIM,
        )
    else:
        evaluation_model = GIN(
            number_of_features,
            config["hidden_dim"],
            config["num_layers"],
            config["dropout"],
        )
    evaluation_model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    evaluation_model.eval()

    records, true_labels, predictions = collect_prediction_records(
        evaluation_model, evaluation_loader, dataset_metadata
    )
    predictions_df = build_predictions_df(records)

    return {
        "model": evaluation_model,
        "dataset": evaluation_dataset,
        "loader": evaluation_loader,
        "predictions_df": predictions_df,
        "accuracy": predictions_df["correct"].mean(),
        "f1": metrics.f1_score(true_labels, predictions, zero_division=0),  # type: ignore[arg-type]
        "config": config,
    }


def load_json(path: str) -> dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def collect_prediction_records(
    model: nn.Module,
    loader: torch_geometric.loader.DataLoader,
    dataset_metadata: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[int], list[int]]:
    records: list[dict[str, Any]] = []
    true_labels: list[int] = []
    predictions: list[int] = []
    sample_idx = 0

    for batch in loader:
        with torch.no_grad():
            logits = model(batch).view(-1)
            probs = logits.sigmoid()
            pred = (logits > 0).float()

        for i, graph in enumerate(batch.to_data_list()):
            true_label = int(graph.y.item())
            pred_label = int(pred[i].item())
            meta = dataset_metadata[sample_idx]
            records.append(
                {
                    "sample_idx": sample_idx,
                    "num_nodes": meta["num_nodes"],
                    "regular": meta["regular"],
                    "paut_size": meta["paut_size"],
                    "aut_grp_order": meta["aut_grp_order"],
                    "true_label": true_label,
                    "prediction": pred_label,
                    "pred_prob": float(probs[i].item()),
                    "correct": pred_label == true_label,
                }
            )
            true_labels.append(true_label)
            predictions.append(pred_label)
            sample_idx += 1

    return records, true_labels, predictions


def build_predictions_df(records: list[dict[str, Any]]) -> pd.DataFrame:
    predictions_df = pd.DataFrame(records)
    predictions_df["paut_relative_size"] = (
        predictions_df["paut_size"] / predictions_df["num_nodes"]
    )
    predictions_df["error"] = (
        predictions_df["true_label"] != predictions_df["prediction"]
    ).astype(int)
    return predictions_df


# ── Analysis helpers ───────────────────────────────────────────────────────────


def plot_confusion_matrix(df, model_name):
    cm = confusion_matrix(df["true_label"], df["prediction"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    print(f"Confusion Matrix — {model_name}")
    plt.show()

    report = classification_report(df["true_label"], df["prediction"], zero_division=0)  # type: ignore[arg-type]
    print(report)


def analyze_regularity(df, model_name):
    wrong = df[~df["correct"]]
    base_rate = df["regular"].mean()
    error_rate = wrong["regular"].mean() if len(wrong) > 0 else 0
    print(f"--- {model_name} ---")
    print(f"Regular graphs in dataset:     {base_rate:.2%}")
    print(f"Regular graphs among errors:   {error_rate:.2%}")

    n_total = len(df)
    n_errors = (~df["correct"]).sum()
    n_regular = df["regular"].sum()
    n_err_reg = ((~df["correct"]) & df["regular"]).sum()
    n_err_nonreg = n_errors - n_err_reg
    n_corr_reg = n_regular - n_err_reg
    n_corr_nonreg = n_total - n_errors - n_corr_reg

    table = np.array([[n_err_reg, n_err_nonreg], [n_corr_reg, n_corr_nonreg]])
    chi2, p, dof, _ = chi2_contingency(table)
    phi = np.sqrt(float(chi2) / n_total)  # type: ignore[arg-type]
    print(f"Chi-squared test:              chi2 = {chi2:.3f}, dof = {dof}, p = {p:.2e}")
    print(f"Effect size (phi):             {phi:.4f}")


def plot_error_by_size(df, model_name):
    error_rate_by_nodes = (
        df.groupby("num_nodes", observed=True)["correct"]
        .agg(error_rate=lambda x: 1 - x.mean(), count="count")
        .reset_index()
    )

    fig, ax1 = plt.subplots(figsize=(10, 5))
    plot_data = error_rate_by_nodes.copy()
    plot_data["num_nodes"] = plot_data["num_nodes"].astype(str)

    ax2 = ax1.twinx()
    sns.barplot(
        data=plot_data,
        x="num_nodes",
        y="count",
        ax=ax2,
        color="gray",
        alpha=0.2,
        zorder=1,
    )
    ax2.set_ylabel("Example Count", color="gray")

    sns.lineplot(
        data=plot_data, x="num_nodes", y="error_rate", ax=ax1, marker="o", zorder=3
    )
    ax1.set_xlabel("Number of Nodes")
    ax1.set_ylabel("Error Rate")
    ax1.set_axisbelow(True)
    ax2.grid(False)

    print(f"Error Rate vs Number of Nodes — {model_name}")
    plt.tight_layout()
    plt.show()


def analyze_feature_importance(df, model_name):
    features = ["num_nodes", "paut_size", "aut_grp_order", "paut_relative_size"]
    X = df[features]
    y = df["error"]

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)

    lr = LogisticRegression().fit(X, y)

    perm = permutation_importance(rf, X, y, n_repeats=30, random_state=42)

    rows = []
    for i, feat in enumerate(features):
        r, p = spearmanr(X[feat], y)
        rows.append(
            {
                "Feature": feat,
                "RF Importance": rf.feature_importances_[i],
                "Permutation Importance": perm["importances_mean"][i],
                "LR Coefficient": lr.coef_[0][i],
                "Spearman r": r,
                "p-value": p,
            }
        )

    result = (
        pd.DataFrame(rows)
        .set_index("Feature")
        .sort_values("Permutation Importance", ascending=False)
    )
    print(f"--- {model_name} ---")
    display(
        result.style.format(
            {
                "RF Importance": "{:.4f}",
                "Permutation Importance": "{:.4f}",
                "LR Coefficient": "{:.4f}",
                "Spearman r": "{:.3f}",
                "p-value": "{:.4f}",
            }
        )
    )
    return result


def plot_error_by_paut_relative_size(df, model_name, ylim=(0, 0.4)):
    lo = df["paut_relative_size"].quantile(0.01)
    hi = df["paut_relative_size"].quantile(0.99)
    bins = np.linspace(lo, hi, 12).tolist()

    df = df.copy()
    df["rel_bin"] = pd.cut(x=df["paut_relative_size"], bins=bins, include_lowest=True)

    grouped = df.groupby("rel_bin", observed=True)["correct"]
    error_rate = grouped.apply(lambda x: 1 - x.mean()).reset_index()
    error_rate.columns = ["rel_bin", "error_rate"]
    error_rate["bin_midpoint"] = [(b.left + b.right) / 2 for b in error_rate["rel_bin"]]

    corr, pvalue = spearmanr(df["paut_relative_size"], df["error"])
    print(f"Spearman r = {corr:.3f}, p = {pvalue:.4f}")

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(
        data=error_rate,
        x="bin_midpoint",
        y="error_rate",
        marker="o",
        color="steelblue",
        ax=ax,
    )
    sns.regplot(
        data=error_rate,
        x="bin_midpoint",
        y="error_rate",
        scatter=False,
        ci=None,
        color="darkred",
        ax=ax,
        order=1,
    )

    ax.set_ylim(*ylim)
    ax.set_xlabel("Relative Partial Automorphism Size")
    ax.set_ylabel("Error Rate")
    ax.set_xticks(error_rate["bin_midpoint"])
    ax.set_xticklabels(
        [f"{x:.2f}" for x in error_rate["bin_midpoint"]], rotation=45, ha="right"
    )
    print(f"Error Rate vs Relative Partial Automorphism Size — {model_name}")
    plt.tight_layout()
    plt.show()


def decode_mapping(graph):
    """Recover the partial automorphism mapping from node features."""
    x = graph.x
    n = graph.num_nodes
    mapping = {}
    for node in range(n):
        encoded = x[node, FEATURE_TARGET_ID].item()
        if encoded != -1:
            target = round(encoded * n)
            mapping[node] = target
    return mapping


def error_by_aut_grp(df, edges):
    d = df.copy()
    d["bucket"] = pd.cut(d["aut_grp_order"], bins=edges, include_lowest=True)
    return (
        d.groupby("bucket", observed=True)["error"]
        .agg(error_rate="mean", count="count")
        .reset_index()
    )


def compute_avg_mapping_distance(graph):
    """Mean shortest-path distance over all (src, f(src)) pairs in the partial automorphism."""
    x = graph.x
    n = graph.num_nodes
    G = torch_geometric.utils.to_networkx(graph, to_undirected=True)

    pairs = []
    for node in range(n):
        encoded = x[node, FEATURE_TARGET_ID].item()
        if encoded != -1:
            target = round(encoded * n)
            pairs.append((node, target))

    if not pairs:
        return np.nan

    distances = []
    for src, tgt in pairs:
        try:
            distances.append(nx.shortest_path_length(G, src, tgt))
        except nx.NetworkXNoPath:
            distances.append(np.nan)

    return float(np.nanmean(distances))


def compute_diameter(graph):
    G = torch_geometric.utils.to_networkx(graph, to_undirected=True)
    if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len))
    return nx.diameter(G)


def can_map_vertex(G, u, v):
    """Return True iff there exists an automorphism of G mapping u to v."""
    if G.degree(u) != G.degree(v):
        return False
    H1, H2 = nx.Graph(G), nx.Graph(G)
    for node in G.nodes():
        H1.nodes[node]["_mark"] = int(node == u)
        H2.nodes[node]["_mark"] = int(node == v)
    GM = nx.algorithms.isomorphism.GraphMatcher(
        H1, H2, node_match=lambda a, b: a["_mark"] == b["_mark"]
    )
    return GM.is_isomorphic()


def is_vertex_transitive(G):
    """Return True iff Aut(G) acts transitively on V(G).

    Checks regularity first as a cheap filter before running isomorphism queries.
    """
    degrees = [d for _, d in G.degree()]
    if len(set(degrees)) > 1:
        return False
    nodes = list(G.nodes())
    v0 = nodes[0]
    return all(can_map_vertex(G, v0, v) for v in nodes[1:])


def has_degree_mismatch(graph):
    """Return True if any (v, f(v)) pair in the partial automorphism has deg(v) != deg(f(v))."""
    x = graph.x
    n = graph.num_nodes
    G = torch_geometric.utils.to_networkx(graph, to_undirected=True)
    deg = dict(G.degree())
    for node in range(n):
        encoded = x[node, FEATURE_TARGET_ID].item()
        if encoded != -1:
            target = round(encoded * n)
            if deg[node] != deg[target]:
                return True
    return False
