# Bachelor Thesis: Graph Neural Network for Automorphism Extension

This repository contains the code, experiments, and validation statistics for my bachelor thesis, which focuses on training a Graph Neural Network (GNN) to determine whether a partial automorphism of a graph can be extended to a full automorphism.

## Overview

The project uses a dataset of graphs with high automorphism group sizes, generates partial automorphisms, and trains a GNN classifier to predict extendability. The goal is to explore the learnability of graph symmetries and the structural patterns that guide automorphism extension.

## Validation Statistics

This section stores training and validation metrics tracked during model development. Additional tables or plots may be added as the experiments evolve.

### Example Metrics

```text
Epoch 03 | Train Loss: 0.4675 | Train Acc: 0.7771 | Train F1: 0.7730 | Val Acc: 0.7680 | Val F1: 0.7647
Epoch 04 | Train Loss: 0.4437 | Train Acc: 0.7842 | Train F1: 0.7948 | Val Acc: 0.7768 | Val F1: ...
```

More detailed logs will be appended as experiments continue.

## Repository Structure

* **/dataset/** — Graph datasets and generated partial automorphisms.
* **/models/** — GNN architectures, training scripts, and saved weights.

## Thesis Context

The goal is to investigate potential applications of graph neural networks (GNNs) to problems in algebraic graph theory. The student will have the opportunity to engage with the state-of-the-art research in machine learning and algebraic graph theory and contribute to the field by generating new record graphs and training GNNs to predict algebraic properties.
