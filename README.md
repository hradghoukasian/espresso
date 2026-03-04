# Espresso-Based Boolean Function Approximation and Decomposition

This repository contains a collection of experiments and heuristics built around Espresso Boolean minimization (via PyEDA). The goal is to study scalability, approximation, and ensemble-style techniques for learning and compressing large Boolean functions represented explicitly as truth tables.

---
## Influence-Based Ensembles and Friedgut's Junta Pipelines

### 🔴  `friedgut_junta_pipeline_partial_tt.py`
Implements a **partial-truth-table** version of the Friedgut-style junta pipeline for settings where the full truth table is unavailable. It samples labeled points from $\\{0,1\\}^B$, estimates influences using **observed neighbor pairs** $(x, x^{ \oplus i})$, learns a junta surrogate from the projected samples, and supports **multi-stage residual learning** to iteratively fit and boost accuracy on incorrectly classified samples.

###  `friedgut_junta_pipeline.py`
Implements an influence-based Boolean function approximation pipeline inspired by Friedgut’s Junta Theorem. It estimates influences, selects top influential variables, constructs a marginalized surrogate function, and applies Espresso minimization.

### `friedgut_junta_pipeline_two_layer.py`
Extends the junta pipeline to a two-layer residual model using XOR boosting. The second stage learns the residual errors of the first-stage predictor and combines predictions via $F \oplus G$ to improve accuracy.

### `ensemble_column_influence_weighted_vote.py`
Implements a column-based ensemble where each subcircuit is built from a random subset of input variables and weighted by the average influence of its selected bits. Final predictions are obtained via influence-weighted voting over projected majority truth tables.

### `ensemble_column_nonzero_influence_weighted_vote.py`
A refinement of the influence-weighted ensemble where subcircuits sample columns only from bits with nonzero influence. This restricts learning to active variables and evaluates whether filtering irrelevant coordinates improves approximation quality.

### `ensemble_column_nonzero_influence_exp_weighted_vote.py`
Extends the nonzero-influence ensemble by applying softmax (temperature-controlled) exponential weighting over circuit scores. This concentrates weight on higher-influence subcircuits and studies sharper attention-style aggregation.

### `ensemble_nonzero_influence_accuracy_exp_weighted.py`
Trains subcircuits on nonzero-influence bits and assigns weights based on each circuit’s full truth-table accuracy using an exponential cutoff rule. This explores accuracy-aware ensemble aggregation instead of influence-based weighting.


  


## Other Files

### `espresso_minimization.py`
Core utilities for running Espresso minimization on Boolean truth tables (with optional don’t-cares) using PyEDA, and reporting statistics such as number of product terms, number of literals, and runtime.

### `benchmark_espresso.py`
A simple benchmarking script that generates random truth tables and measures Espresso runtime across multiple trials to study scalability as the number of input variables grows.

### `run_examples.py`
Minimal example script demonstrating how to (i) minimize a randomly generated truth table and (ii) construct a truth table from explicit ON/OFF sets before applying Espresso.

### `batch_espresso.py`
Implements a two-stage “batch Espresso” heuristic: first compress a subset of variables into an intermediate Boolean function via majority aggregation, then learn a second-stage function and evaluate the composed approximation.

### `Shanon_decomposition.py`
A variant of Shannon decomposition configured for larger selector splits (e.g., \(k=10\)), with truth-table slicing aligned with PyEDA’s indexing convention for efficient block processing.

### `shanon_vs_flat_espresso.py`
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
