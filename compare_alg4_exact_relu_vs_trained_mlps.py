"""

Experiment:
  1) Generate a random S-junta f:{0,1}^B -> {0,1}.
  2) Sample a partial truth table D_train and a test set D_test.
  3) Run Algorithm 3: multi-stage influence-based residual learning with Espresso.
  4) Convert the learned Algorithm-3 predictor into an exact ReLU MLP
     (Algorithm-4-style explicit Boolean-gate-to-ReLU construction).
  5) Train a standard ReLU MLP and a standard Sigmoid MLP on the same D_train,
     using the same number of hidden layers and the same hidden widths as the
     compiled exact ReLU network.
  6) Compare test accuracy and training/construction runtime over seeds.

Notes:
  - Training time reported for Algorithm 4 includes Algorithm 3 training plus
    exact ReLU construction time, and excludes test evaluation time.
  - Training time reported for trainable MLPs excludes test evaluation time.

"""

from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception as exc:  # pragma: no cover
    torch = None
    nn = None
    F = None
    TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None

try:
    from pyeda.inter import exprvars, truthtable
    from pyeda.boolalg.minimization import espresso_tts
except Exception as exc:  # pragma: no cover
    exprvars = None
    truthtable = None
    espresso_tts = None
    PYEDA_IMPORT_ERROR = exc
else:
    PYEDA_IMPORT_ERROR = None


# -----------------------------------------------------------------------------
# Requirements
# -----------------------------------------------------------------------------

def require_pyeda() -> None:
    if PYEDA_IMPORT_ERROR is not None:
        raise RuntimeError(
            "PyEDA is required for Espresso calls. Install it with:\n"
            "    pip install pyeda\n"
            f"Original import error: {PYEDA_IMPORT_ERROR!r}"
        )


def require_torch() -> None:
    if TORCH_IMPORT_ERROR is not None:
        raise RuntimeError(
            "PyTorch is required for the MLP baselines. Install it with:\n"
            "    pip install torch\n"
            f"Original import error: {TORCH_IMPORT_ERROR!r}"
        )


# -----------------------------------------------------------------------------
# Basic Boolean utilities
# -----------------------------------------------------------------------------

def proj_index(x: int, cols: Sequence[int]) -> int:
    """Project integer-coded bitstring x to coordinates cols, packed as int."""
    u = 0
    for j, bit in enumerate(cols):
        if (x >> bit) & 1:
            u |= 1 << j
    return u


@dataclass(frozen=True)
class SJunta:
    B: int
    junta_bits: List[int]
    junta_tt: List[int]

    def __call__(self, x: int) -> int:
        return self.junta_tt[proj_index(x, self.junta_bits)]


def make_random_s_junta(B: int, S: int, rng: random.Random) -> SJunta:
    if not (0 <= S <= B):
        raise ValueError("Need 0 <= S <= B.")
    junta_bits = sorted(rng.sample(range(B), S))
    junta_tt = [rng.getrandbits(1) for _ in range(1 << S)]
    return SJunta(B=B, junta_bits=junta_bits, junta_tt=junta_tt)


def sample_dataset(f, B: int, T: int, rng: random.Random) -> Tuple[List[int], List[int]]:
    """Uniform sample without replacement from {0,1}^B."""
    cube_size = 1 << B
    if T > cube_size:
        raise ValueError(f"T={T} exceeds cube size 2^B={cube_size}.")
    xs = rng.sample(range(cube_size), T)
    ys = [int(f(x)) for x in xs]
    return xs, ys


def full_cube_dataset(f, B: int) -> Tuple[List[int], List[int]]:
    xs = list(range(1 << B))
    ys = [int(f(x)) for x in xs]
    return xs, ys


def ints_to_bit_matrix(xs: Sequence[int], B: int) -> np.ndarray:
    """Convert integer-coded Boolean inputs to an array of shape (n, B)."""
    arr = np.asarray(xs, dtype=np.uint64)
    bits = ((arr[:, None] >> np.arange(B, dtype=np.uint64)) & 1).astype(np.float32)
    return bits


def accuracy_from_int_predictor(predict_fn, xs: Sequence[int], ys: Sequence[int]) -> float:
    correct = 0
    for x, y in zip(xs, ys):
        correct += int(int(predict_fn(int(x))) == int(y))
    return correct / max(1, len(xs))


# -----------------------------------------------------------------------------
# Espresso wrapper and Algorithm 3
# -----------------------------------------------------------------------------

@dataclass
class EspressoModel:
    n_vars: int
    expr: object
    constant_value: Optional[int] = None
    tt_full: Optional[List[int]] = None

    def predict_one(self, x: int) -> int:
        if self.tt_full is not None:
            return int(self.tt_full[x])
        if self.constant_value is not None:
            return int(self.constant_value)
        X = exprvars("x", self.n_vars)
        assignment = {X[j]: ((x >> j) & 1) for j in range(self.n_vars)}
        val = self.expr.restrict(assignment)
        return 1 if val.is_one() else 0


def espresso_learn_tt(tt: Sequence[str], n_vars: int) -> EspressoModel:
    """Run Espresso on a truth table over {'0','1','-'} and cache full predictions."""
    require_pyeda()
    if len(tt) != (1 << n_vars):
        raise ValueError(f"len(tt)={len(tt)} but expected 2^{n_vars}.")
    chars = list(tt)
    values = set(chars)
    if not values.issubset({"0", "1", "-", "x"}):
        raise ValueError(f"Truth table has invalid characters: {values}.")

    specified = {c for c in chars if c in {"0", "1"}}
    if n_vars == 0:
        val = 1 if chars[0] == "1" else 0
        return EspressoModel(n_vars=0, expr=None, constant_value=val, tt_full=[val])
    if specified == set() or specified == {"0"}:
        return EspressoModel(n_vars=n_vars, expr=None, constant_value=0, tt_full=[0] * (1 << n_vars))
    if specified == {"1"}:
        return EspressoModel(n_vars=n_vars, expr=None, constant_value=1, tt_full=[1] * (1 << n_vars))

    X = exprvars("x", n_vars)
    f_tt = truthtable(X, "".join(chars))
    expr, = espresso_tts(f_tt)

    # Cache the learned total function on all projected inputs. This makes
    # Algorithm-3 prediction and exact ReLU compilation fast and deterministic.
    full = []
    for u in range(1 << n_vars):
        assignment = {X[j]: ((u >> j) & 1) for j in range(n_vars)}
        val = expr.restrict(assignment)
        full.append(1 if val.is_one() else 0)
    return EspressoModel(n_vars=n_vars, expr=expr, tt_full=full)


@dataclass
class Stage:
    bits: List[int]
    tt_full: List[int]

    def predict_one(self, x: int) -> int:
        return int(self.tt_full[proj_index(x, self.bits)])


def predict_algorithm3(stages: Sequence[Stage], x: int) -> int:
    y = 0
    for st in stages:
        y ^= st.predict_one(x)
    return y


def influences_from_observed_pairs(
    B: int,
    x_train: Sequence[int],
    residual_map: Dict[int, int],
) -> Tuple[List[float], List[int]]:
    """Algorithm-3 influence estimator using observed Hamming-neighbor pairs."""
    xset = set(x_train)
    influences = [0.0] * B
    pair_counts = [0] * B
    for i in range(B):
        mask = 1 << i
        mismatches = 0
        count = 0
        for x in x_train:
            xn = x ^ mask
            if xn in xset:
                count += 1
                mismatches += int(residual_map[x] != residual_map[xn])
        pair_counts[i] = count
        influences[i] = (mismatches / count) if count else 0.0
    return influences, pair_counts


def select_topk_above_tau(influences: Sequence[float], K: int, tau: float, rng: random.Random) -> List[int]:
    candidates = [i for i, v in enumerate(influences) if v > tau]
    rng.shuffle(candidates)  # random tie-breaking before sorting
    candidates.sort(key=lambda i: influences[i], reverse=True)
    return candidates[: min(K, len(candidates))]


def projected_residual_partial_tt(
    x_train: Sequence[int],
    r_train: Sequence[int],
    J: Sequence[int],
    rng: random.Random,
) -> List[str]:
    """Empirical majority table on J; unseen patterns are don't-cares '-'."""
    k = len(J)
    if k == 0:
        ones = sum(r_train)
        zeros = len(r_train) - ones
        if ones > zeros:
            return ["1"]
        if zeros > ones:
            return ["0"]
        return [str(rng.getrandbits(1))]

    size = 1 << k
    zeros = [0] * size
    ones = [0] * size
    for x, r in zip(x_train, r_train):
        u = proj_index(x, J)
        if r:
            ones[u] += 1
        else:
            zeros[u] += 1

    out: List[str] = []
    for u in range(size):
        if zeros[u] == 0 and ones[u] == 0:
            out.append("-")
        elif ones[u] > zeros[u]:
            out.append("1")
        elif zeros[u] > ones[u]:
            out.append("0")
        else:
            out.append(str(rng.getrandbits(1)))
    return out


def train_algorithm3(
    B: int,
    K: int,
    tau: float,
    stages_m: int,
    x_train: Sequence[int],
    y_train: Sequence[int],
    seed: int,
) -> Tuple[List[Stage], float]:
    """Train Algorithm 3. Return learned stages and pure training runtime."""
    rng = random.Random(seed)
    stages: List[Stage] = []
    cumulative_train_time = 0.0

    for _t in range(1, stages_m + 1):
        stage_start = time.perf_counter()

        r_train: List[int] = []
        residual_map: Dict[int, int] = {}
        for x, y in zip(x_train, y_train):
            r = int(y) ^ predict_algorithm3(stages, x)
            r_train.append(r)
            residual_map[x] = r

        infl, _pair_counts = influences_from_observed_pairs(B, x_train, residual_map)
        J = select_topk_above_tau(infl, K=K, tau=tau, rng=rng)
        tt_proj = projected_residual_partial_tt(x_train, r_train, J, rng=rng)
        model = espresso_learn_tt(tt_proj, n_vars=len(J))
        stages.append(Stage(bits=list(J), tt_full=list(model.tt_full or [])))

        cumulative_train_time += time.perf_counter() - stage_start

    return stages, cumulative_train_time


# -----------------------------------------------------------------------------
# Algorithm-4-style exact ReLU construction from the learned stages
# -----------------------------------------------------------------------------

class ExactReluFromAlgorithm3(nn.Module):
    """
    Exact ReLU realization of the learned Algorithm-3 Boolean predictor.

    Construction:
      - Layer 1: one ReLU minterm neuron for each 1-entry in each stage truth table.
      - Layer 2: sums minterms belonging to each stage, giving stage outputs.
      - Layer 3/output: ReLU piecewise-linear parity of the sum of stage outputs.

    This network is intended for Boolean inputs. It agrees with Algorithm 3 on
    {0,1}^B up to numerical precision.
    """

    def __init__(self, B: int, stages: Sequence[Stage]):
        super().__init__()
        self.B = B
        self.num_stages = len(stages)

        W1_rows: List[np.ndarray] = []
        b1_vals: List[float] = []
        minterm_stage: List[int] = []

        for t, st in enumerate(stages):
            k = len(st.bits)
            for u, val in enumerate(st.tt_full):
                if int(val) != 1:
                    continue
                w = np.zeros(B, dtype=np.float32)
                num_zero_literals = 0
                for j, bit in enumerate(st.bits):
                    if (u >> j) & 1:
                        w[bit] += 1.0
                    else:
                        w[bit] -= 1.0
                        num_zero_literals += 1
                # AND(l_1,...,l_k) = ReLU(sum literals - (k-1)).
                # sum literals = w^T x + num_zero_literals.
                if k == 0:
                    b = 1.0  # constant-one minterm
                else:
                    b = float(num_zero_literals - (k - 1))
                W1_rows.append(w)
                b1_vals.append(b)
                minterm_stage.append(t)

        num_minterms = len(W1_rows)
        self.num_minterms = num_minterms

        if num_minterms == 0:
            W1 = np.zeros((1, B), dtype=np.float32)
            b1 = np.zeros(1, dtype=np.float32)
            W2 = np.zeros((max(1, self.num_stages), 1), dtype=np.float32)
        else:
            W1 = np.stack(W1_rows, axis=0).astype(np.float32)
            b1 = np.asarray(b1_vals, dtype=np.float32)
            W2 = np.zeros((max(1, self.num_stages), num_minterms), dtype=np.float32)
            for m, t in enumerate(minterm_stage):
                W2[t, m] = 1.0

        self.register_buffer("W1", torch.tensor(W1))
        self.register_buffer("b1", torch.tensor(b1))
        self.register_buffer("W2", torch.tensor(W2))

        # Hidden widths used for capacity-matched trainable baselines.
        # Layer 1 = minterms, layer 2 = stage outputs, layer 3 = parity ReLU thresholds.
        parity_width = max(1, self.num_stages - 1)
        self.hidden_widths = [max(1, num_minterms), max(1, self.num_stages), parity_width]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (n, B), Boolean-valued for exactness.
        h1 = F.relu(x @ self.W1.T + self.b1)
        stage_out = h1 @ self.W2.T

        if self.num_stages == 0:
            return torch.zeros((x.shape[0],), device=x.device)
        if self.num_stages == 1:
            return stage_out[:, 0]

        s = stage_out.sum(dim=1)
        out = s.clone()
        # parity(s) = s + sum_{j=1}^{m-1} 2*(-1)^j ReLU(s-j)
        # on integer s in {0,...,m}.
        for j in range(1, self.num_stages):
            out = out + (2.0 * ((-1.0) ** j)) * F.relu(s - float(j))
        return out


@torch.no_grad()
def exact_relu_accuracy(model: ExactReluFromAlgorithm3, xs: Sequence[int], ys: Sequence[int], B: int, batch_size: int = 8192) -> float:
    model.eval()
    y_np = np.asarray(ys, dtype=np.int64)
    correct = 0
    n = len(xs)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        xb = torch.tensor(ints_to_bit_matrix(xs[start:end], B), dtype=torch.float32)
        out = model(xb)
        pred = (out >= 0.5).long().cpu().numpy()
        correct += int((pred == y_np[start:end]).sum())
    return correct / max(1, n)


# -----------------------------------------------------------------------------
# Trainable MLP baselines
# -----------------------------------------------------------------------------

class TrainableMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_widths: Sequence[int], activation: str):
        super().__init__()
        if activation not in {"relu", "sigmoid"}:
            raise ValueError("activation must be 'relu' or 'sigmoid'.")
        self.activation = activation
        widths = [input_dim] + [int(max(1, w)) for w in hidden_widths] + [1]
        layers: List[nn.Module] = []
        for i in range(len(widths) - 1):
            layers.append(nn.Linear(widths[i], widths[i + 1]))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.layers[:-1]:
            h = layer(h)
            if self.activation == "relu":
                h = F.relu(h)
            else:
                h = torch.sigmoid(h)
        return self.layers[-1](h).squeeze(-1)  # logits


@torch.no_grad()
def mlp_accuracy(model: nn.Module, xs: Sequence[int], ys: Sequence[int], B: int, batch_size: int = 8192) -> float:
    model.eval()
    y_np = np.asarray(ys, dtype=np.int64)
    correct = 0
    n = len(xs)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        xb = torch.tensor(ints_to_bit_matrix(xs[start:end], B), dtype=torch.float32)
        logits = model(xb)
        pred = (torch.sigmoid(logits) >= 0.5).long().cpu().numpy()
        correct += int((pred == y_np[start:end]).sum())
    return correct / max(1, n)


def train_mlp_baseline(
    B: int,
    hidden_widths: Sequence[int],
    activation: str,
    x_train: Sequence[int],
    y_train: Sequence[int],
    x_test: Sequence[int],
    y_test: Sequence[int],
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
) -> Tuple[float, float]:
    """Train a ReLU or Sigmoid MLP. Return (test_acc, training_time_s)."""
    require_torch()
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32 - 1))
    random.seed(seed)

    model = TrainableMLP(B, hidden_widths, activation=activation)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    X = torch.tensor(ints_to_bit_matrix(x_train, B), dtype=torch.float32)
    y = torch.tensor(np.asarray(y_train, dtype=np.float32), dtype=torch.float32)
    n = X.shape[0]

    start_time = time.perf_counter()
    model.train()
    for _epoch in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            xb = X[idx]
            yb = y[idx]
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
    train_time = time.perf_counter() - start_time

    acc = mlp_accuracy(model, x_test, y_test, B)
    return acc, train_time


# -----------------------------------------------------------------------------
# Reporting helpers
# -----------------------------------------------------------------------------

def mean_std(vals: Iterable[float]) -> Tuple[float, float]:
    vals = [float(v) for v in vals if float(v) == float(v)]
    if not vals:
        return float("nan"), float("nan")
    if len(vals) == 1:
        return vals[0], 0.0
    return statistics.mean(vals), statistics.stdev(vals)


def fmt_pm(mean: float, std: float, digits: int = 3) -> str:
    if mean != mean:
        return "NA"
    return f"{mean:.{digits}f} +- {std:.{digits}f}"


def summarize_method(method: str, rows: List[Dict[str, object]], acc_key: str, time_key: str) -> Dict[str, object]:
    acc_mu, acc_sd = mean_std(r[acc_key] for r in rows)
    t_mu, t_sd = mean_std(r[time_key] for r in rows)
    return {
        "method": method,
        "test_accuracy_mean": acc_mu,
        "test_accuracy_std": acc_sd,
        "test_accuracy_mean_pm_std": fmt_pm(acc_mu, acc_sd),
        "runtime_s_mean": t_mu,
        "runtime_s_std": t_sd,
        "runtime_s_mean_pm_std": fmt_pm(t_mu, t_sd, digits=2),
    }


# -----------------------------------------------------------------------------
# Main experiment
# -----------------------------------------------------------------------------

def run_experiment(args: argparse.Namespace) -> Path:
    require_pyeda()
    require_torch()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    per_seed_rows: List[Dict[str, object]] = []

    for s in range(args.num_seeds):
        seed = args.base_seed + s
        rng = random.Random(seed)
        f = make_random_s_junta(args.B, args.S, rng)
        x_train, y_train = sample_dataset(f, args.B, args.train_size, rng)
        if args.full_cube_test:
            x_test, y_test = full_cube_dataset(f, args.B)
        else:
            x_test, y_test = sample_dataset(f, args.B, args.test_size, rng)

        print(f"[seed {seed}] junta_bits={f.junta_bits}")

        # Algorithm 3 training.
        stages, alg3_train_time = train_algorithm3(
            B=args.B,
            K=args.K,
            tau=args.tau,
            stages_m=args.stages,
            x_train=x_train,
            y_train=y_train,
            seed=seed,
        )
        alg3_acc = accuracy_from_int_predictor(lambda z: predict_algorithm3(stages, z), x_test, y_test)

        # Algorithm 4 exact ReLU construction.
        compile_start = time.perf_counter()
        exact_relu = ExactReluFromAlgorithm3(B=args.B, stages=stages)
        compile_time = time.perf_counter() - compile_start
        exact_acc = exact_relu_accuracy(exact_relu, x_test, y_test, args.B)
        exact_total_time = alg3_train_time + compile_time
        agreement_gap = abs(exact_acc - alg3_acc)

        hidden_widths = exact_relu.hidden_widths
        total_hidden_neurons = int(sum(hidden_widths))
        depth_hidden_layers = len(hidden_widths)

        # Train matched ReLU and Sigmoid MLP baselines.
        relu_acc, relu_time = train_mlp_baseline(
            B=args.B,
            hidden_widths=hidden_widths,
            activation="relu",
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            seed=10_000 + seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        sig_acc, sig_time = train_mlp_baseline(
            B=args.B,
            hidden_widths=hidden_widths,
            activation="sigmoid",
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            seed=20_000 + seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        row = {
            "seed": seed,
            "B": args.B,
            "S": args.S,
            "K": args.K,
            "tau": args.tau,
            "stages": args.stages,
            "train_size": args.train_size,
            "test_size": len(x_test),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "junta_bits": " ".join(map(str, f.junta_bits)),
            "hidden_widths": " ".join(map(str, hidden_widths)),
            "depth_hidden_layers": depth_hidden_layers,
            "total_hidden_neurons": total_hidden_neurons,
            "alg3_test_acc": alg3_acc,
            "alg3_train_time_s": alg3_train_time,
            "alg4_exact_relu_test_acc": exact_acc,
            "alg4_compile_time_s": compile_time,
            "alg4_total_construction_time_s": exact_total_time,
            "alg4_agreement_gap_vs_alg3": agreement_gap,
            "trained_relu_mlp_test_acc": relu_acc,
            "trained_relu_mlp_train_time_s": relu_time,
            "trained_sigmoid_mlp_test_acc": sig_acc,
            "trained_sigmoid_mlp_train_time_s": sig_time,
        }
        per_seed_rows.append(row)

        print(
            f"  Algorithm 3 / exact ReLU: acc={exact_acc:.4f}, "
            f"time={exact_total_time:.2f}s, widths={hidden_widths}, "
            f"agreement_gap={agreement_gap:.2e}"
        )
        print(f"  Trained ReLU MLP      : acc={relu_acc:.4f}, train_time={relu_time:.2f}s")
        print(f"  Trained Sigmoid MLP   : acc={sig_acc:.4f}, train_time={sig_time:.2f}s")

    # Summary.
    summary_rows: List[Dict[str, object]] = []
    base_info = {
        "B": args.B,
        "S-junta": args.S,
        "K": args.K,
        "stages": args.stages,
        "train_size": args.train_size,
        "test_size": per_seed_rows[0]["test_size"] if per_seed_rows else args.test_size,
        "num_seeds": args.num_seeds,
        "epochs": args.epochs,
    }

    for item in [
        summarize_method(
            "Algorithm 4 exact ReLU (compiled from Algorithm 3)",
            per_seed_rows,
            "alg4_exact_relu_test_acc",
            "alg4_total_construction_time_s",
        ),
        summarize_method(
            "Trained ReLU MLP (matched architecture)",
            per_seed_rows,
            "trained_relu_mlp_test_acc",
            "trained_relu_mlp_train_time_s",
        ),
        summarize_method(
            "Trained Sigmoid MLP (matched architecture)",
            per_seed_rows,
            "trained_sigmoid_mlp_test_acc",
            "trained_sigmoid_mlp_train_time_s",
        ),
    ]:
        row = dict(base_info)
        row.update(item)
        width_mu, width_sd = mean_std(r["total_hidden_neurons"] for r in per_seed_rows)
        row["total_hidden_neurons_mean_pm_std"] = fmt_pm(width_mu, width_sd, digits=1)
        summary_rows.append(row)

    summary_path = outdir / "alg4_exact_relu_vs_trained_mlps_summary.csv"
    with summary_path.open("w", newline="") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    if args.save_per_seed:
        per_seed_path = outdir / "alg4_exact_relu_vs_trained_mlps_per_seed.csv"
        with per_seed_path.open("w", newline="") as fcsv:
            writer = csv.DictWriter(fcsv, fieldnames=list(per_seed_rows[0].keys()))
            writer.writeheader()
            writer.writerows(per_seed_rows)
        print(f"Saved per-seed results to: {per_seed_path}")

    print("\n=== Summary: mean +- sample std over seeds ===")
    header = f"{'Method':52s} | {'Test acc':17s} | {'Runtime [s]':17s}"
    print(header)
    print("-" * len(header))
    for r in summary_rows:
        print(f"{str(r['method']):52s} | {r['test_accuracy_mean_pm_std']:17s} | {r['runtime_s_mean_pm_std']:17s}")

    print(f"\nSaved summary table to: {summary_path}")
    return summary_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--B", type=int, default=21, help="ambient input dimension")
    p.add_argument("--S", type=int, default=10, help="number of relevant variables in random S-junta")
    p.add_argument("--train-size", type=int, default=2**21)
    p.add_argument("--test-size", type=int, default=2**17, help="ignored when --full-cube-test is used")
    p.add_argument("--full-cube-test", action="store_true", help="evaluate on all 2^B Boolean inputs")
    p.add_argument("--K", type=int, default=8, help="top-K selected variables per residual stage")
    p.add_argument("--tau", type=float, default=0.0, help="influence threshold")
    p.add_argument("--stages", type=int, default=20, help="number of residual stages")
    p.add_argument("--num-seeds", type=int, default=20)
    p.add_argument("--base-seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=1, help="training epochs for trainable MLP baselines")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--save-per-seed", action="store_true")
    p.add_argument("--outdir", type=str, default="results_alg4_vs_mlp")
    return p.parse_args()


if __name__ == "__main__":
    run_experiment(parse_args())
