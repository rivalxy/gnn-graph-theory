# Bachelor Thesis: Graph neural networks and extensibility of partial graph automorphisms

This repository contains the code, experiments, and validation statistics for my bachelor thesis, which focuses on training a Graph Neural Network (GNN) to determine whether a partial automorphism of a graph can be extended to a full automorphism.

## Overview

The project uses a dataset of graphs with high automorphism group sizes, generates partial automorphisms, and trains a GNN classifier to predict extendibility. The goal is to explore the learnability of graph symmetries and the structural patterns that guide automorphism extension.

## Repository Structure

* **`models.py`** - GNN model definitions (GIN and GPS architectures)
* **`utils.py`** - Training helpers, evaluation, and metadata utilities
* **`/dataset/`** - Graph datasets, generation scripts, and dataset statistics notebook
* **`/kaggle/`** - Notebooks used for training on kaggle.com (baseline, 7_features, gps variants)
* **`/results/`** - Optuna search history, training history, and best model weights (`.pt` files)
* **`/tests/`** - Unit tests (pytest)

## Installation

This project uses [uv](https://github.com/astral-sh/uv) and requires Python 3.13.

```bash
uv sync
```

Or install dependencies manually with pip using `pyproject.toml`.

Key dependencies: `torch`, `torch-geometric`, `pynauty`, `networkx`, `optuna`, `scikit-learn`, `matplotlib`.

## Setup

### 1. Generate datasets

If the dataset `.pt` files are not present, generate them from the source graphs:

```bash
python dataset/generate.py
```

This reads `dataset/all_graphs.g6`, splits graphs into train/val/test (80/10/10), generates partial automorphism examples with extendibility labels, and saves `dataset/{train,val,test}_dataset.pt`.

### 2. Add positional encodings (GPS models only)

Before training or evaluating GPS-based models, generate the Laplacian eigenvector positional encodings. The pre-transformed files are excluded from the repository due to their large size.

```bash
python dataset/pe_transform.py
```

This reads `dataset/{train,val,test}_dataset.pt`, adds Laplacian eigenvector PE (dim=5), and saves `dataset/{train,val,test}_dataset_with_pe.pt`.

## Models

### GIN (Graph Isomorphism Network)

Based on [Xu et al., ICLR 2019](https://arxiv.org/abs/1810.00826). Uses an MLP with batch normalization between layers and a jumping-knowledge readout over all layers. Does not require positional encodings.

Two feature variants are available:

* **baseline** — 3 node features: node ID, source mapping target, target mapping source
* **7_features** — extends baseline with degree, clustering coefficient, triangle count, and average neighbor degree

### GPS (General, Powerful, Scalable graph transformer)

Combines local GINConv message passing with global multi-head attention. Requires Laplacian eigenvector PE (5-dim, produced by `pe_transform.py`). PE is projected to `hidden_dim` before being added to node features.

## Training

Training is done via Kaggle notebooks located in `/kaggle/` (subdirectories: `baseline/`, `7_features/`, `gin_lappe/`, `gps/`). Hyperparameter search is performed with Optuna; best configs and trained weights are saved to `/results/`.

## Evaluation

Open and run `evaluate.ipynb` to compare trained models. It loads checkpoints from `/results/`, plots training curves, and reports accuracy and F1 scores on the test set.

For deeper analysis (confusion matrices, feature importance, breakdown by graph properties), use `analysis.ipynb`.

## Testing

```bash
pytest
```

## Thesis Context

The goal is to investigate potential applications of graph neural networks (GNNs) to problems in algebraic graph theory. The student will have the opportunity to engage with the state-of-art research in machine learning and algebraic graph theory and contribute to the field by training GNNs to predict algebraic properties.
