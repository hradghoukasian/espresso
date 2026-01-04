# Espresso-Based Boolean Function Approximation and Decomposition

This repository contains a collection of experiments and heuristics built around **Espresso Boolean minimization** (via PyEDA). The goal is to study scalability, approximation, and ensemble-style techniques for learning and compressing large Boolean functions represented as truth tables.

---

## File Overview

### `espresso_minimization.py`
Core utilities for running Espresso minimization on Boolean truth tables (with optional don’t-cares) using PyEDA, and reporting statistics such as number of product terms, number of literals, and runtime.

### `benchmark_espresso.py`
A simple benchmarking script that generates random truth tables and measures Espresso runtime across multiple trials to study scalability as the number of input variables grows.

### `run_examples.py`
Minimal example script demonstrating how to (i) minimize a randomly generated truth table and (ii) construct a truth table from explicit ON/OFF sets before applying Espresso.

### `batch_esresso.py`
Implements a two-stage “batch Espresso” heuristic: first compress a subset of variables into an intermediate Boolean function via majority aggregation, then learn a second-stage function and evaluate the composed approximation.

### `Shanon_decomposition.py`
Implements Shannon (hierarchical) decomposition by partitioning a large truth table into cofactors, minimizing each cofactor independently with Espresso, and recombining them using selector variables to obtain an exact representation.

### `Shanon_decomposition_k10.py`
A variant of Shannon decomposition configured for larger selector splits (e.g., \(k=10\)), with truth-table slicing aligned with PyEDA’s indexing convention for efficient block processing.

### `Shanon_vs_Normal.py`
Compares hierarchical (Shannon-split) Espresso minimization against a single flat Espresso run in terms of runtime and resulting SOP size, while verifying exact reconstruction accuracy.

### `ensemble_column_unweighted_vote.py`
Implements column-based bagging: randomly samples subsets of input variables, builds reduced subcircuits by majority over marginalized variables, and aggregates predictions using an unweighted majority vote.

### `ensemble_column_weighted_vote.py`
Extends column-based bagging by weighting each subcircuit according to its accuracy on the full truth table, using a centered nonnegative weight and performing weighted majority voting at inference time.

### `row_partition_softmin_vote.py`
Implements row-partitioned experts: trains multiple Espresso circuits on disjoint subsets of rows (others treated as don’t-cares), computes softmin-based attention weights using nearest-row Hamming distances, and aggregates predictions from selected experts.

---

## Notes

- All experiments assume Boolean functions represented explicitly as truth tables.
- Shannon decomposition experiments achieve **exact accuracy** but trade off runtime for memory and circuit size.
- Ensemble and partition-based methods trade exactness for scalability and approximation quality.

---
