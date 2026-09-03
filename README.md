# ☕ Macchiato

Code for the paper:

**Certifiably-Interpretable Training of ReLU-MLPs for Boolean Tasks with Guaranteed Truth-Table Generalization**  
Hrad Ghoukasian and Anastasis Kratsios

This repository contains the experimental implementation of **Macchiato**, our multi-stage Boolean learning procedure based on influence-guided variable selection, low-dimensional Espresso minimization, XOR residual updates, and exact compilation into a ReLU-MLP.

## 📁 Files

- `compare_algorithm1_vs_flat_espresso.py`  
  Compares Macchiato with flat ambient-dimensional Espresso and produces the accuracy and runtime results reported in the paper.

- `algorithm1_over_stages.py`  
  Evaluates Macchiato across successive residual stages.

- `algorithm1_over_stages_plot.py`  
  Generates the accuracy-versus-stage figures.

- `compare_alg1_influence_vs_randomK.py`  
  Produces the influence-selection versus random $K$ ablation.

- `compare_alg2_exact_relu_vs_trained_mlps.py`  
  Compares the exact circuit-derived ReLU realization with gradient-trained ReLU and sigmoid MLPs.

- `espresso_minimization.py`  
  Utility functions for constructing partial truth tables and running Espresso through PyEDA.

## ⚙️ Installation

```bash
git clone https://github.com/hradghoukasian/espresso.git
cd espresso
pip install numpy matplotlib pyeda torch
```

## 🧪 Experiments

The experiments use randomly generated Boolean $S$-junta targets. Unless stated otherwise in the paper, Macchiato uses $\tau=0$, a maximum stage budget $m=20$, and the projection budget $K$ specified by each configuration.

For example:

```bash
python compare_algorithm1_vs_flat_espresso.py \
    --B 15 \
    --S 8 \
    --train-size 4000 \
    --test-size 32768 \
    --K 6 \
    --tau 0 \
    --stages 20 \
    --num-seeds 20
```

## 📄 Citation

```bibtex
@article{ghoukasian2026certifiably,
  title   = {Certifiably-Interpretable Training of ReLU-MLPs for Boolean Tasks with Guaranteed Truth-Table Generalization},
  author  = {Ghoukasian, Hrad and Kratsios, Anastasis},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2026}
}
```
