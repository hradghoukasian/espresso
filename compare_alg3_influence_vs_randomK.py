"""Experiment:
  Compare Algorithm 3 with its random-K ablation.

Methods:
  1) Algorithm 3 (influence top-K):
       At each residual stage, estimate influences from observed neighbor pairs
       in D_train, select the top-K coordinates above tau, build a projected
       partial truth table, and run Espresso.

  2) Algorithm 3 (random K):
       Same residual-learning pipeline, but at each stage choose K coordinates
       uniformly at random from the ambient B variables, instead of selecting
       the top-K influence coordinates.

For each seed, both methods use the same target S-junta, the same training set,
and the same test set. The reported training runtime excludes test evaluation.


"""

from __future__ import annotations

import argparse
import csv
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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
# Basic utilities
# -----------------------------------------------------------------------------

def require_pyeda() -> None:
    if PYEDA_IMPORT_ERROR is not None:
        raise RuntimeError(
            "PyEDA is required for the actual Espresso calls. Install it with:\n"
            "    pip install pyeda\n"
            f"Original import error: {PYEDA_IMPORT_ERROR!r}"
        )


def proj_index(x: int, cols: Sequence[int]) -> int:
    """Project integer-coded bitstring x to the coordinates cols, packed as an int."""
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


def accuracy_from_predictor(predict_fn, xs: Sequence[int], ys: Sequence[int]) -> float:
    if not xs:
        return 0.0
    correct = 0
    for x, y in zip(xs, ys):
        correct += int(int(predict_fn(x)) == int(y))
    return correct / len(xs)


# -----------------------------------------------------------------------------
# Espresso wrapper for 0/1/- truth tables
# -----------------------------------------------------------------------------

@dataclass
class EspressoModel:
    n_vars: int
    expr: object
    constant_value: Optional[int] = None

    def predict_one(self, x: int) -> int:
        if self.constant_value is not None:
            return int(self.constant_value)
        X = exprvars("x", self.n_vars)
        assignment = {X[j]: ((x >> j) & 1) for j in range(self.n_vars)}
        val = self.expr.restrict(assignment)
        return 1 if val.is_one() else 0


def espresso_learn_tt(tt: Sequence[str], n_vars: int) -> EspressoModel:
    """Run Espresso on a truth table over {'0','1','-'} and return a predictor."""
    require_pyeda()
    if len(tt) != (1 << n_vars):
        raise ValueError(f"len(tt)={len(tt)} but expected 2^{n_vars}.")
    chars = list(tt)
    values = set(chars)
    if not values.issubset({"0", "1", "-", "x"}):
        raise ValueError(f"Truth table has invalid characters: {values}.")

    # Degenerate constants/all-DC. This avoids edge cases in PyEDA.
    specified = {c for c in chars if c in {"0", "1"}}
    if n_vars == 0:
        return EspressoModel(n_vars=0, expr=None, constant_value=1 if chars[0] == "1" else 0)
    if specified == set():
        return EspressoModel(n_vars=n_vars, expr=None, constant_value=0)
    if specified == {"0"}:
        return EspressoModel(n_vars=n_vars, expr=None, constant_value=0)
    if specified == {"1"}:
        return EspressoModel(n_vars=n_vars, expr=None, constant_value=1)

    X = exprvars("x", n_vars)
    f_tt = truthtable(X, "".join(chars))
    expr, = espresso_tts(f_tt)
    return EspressoModel(n_vars=n_vars, expr=expr)


# -----------------------------------------------------------------------------
# Algorithm 3 and random-K ablation
# -----------------------------------------------------------------------------

@dataclass
class Stage:
    bits: List[int]
    model: EspressoModel

    def predict_one(self, x: int) -> int:
        if not self.bits:
            return self.model.predict_one(0)
        return self.model.predict_one(proj_index(x, self.bits))


def predict_stages(stages: Sequence[Stage], x: int) -> int:
    y = 0
    for st in stages:
        y ^= st.predict_one(x)
    return y


def influences_from_observed_pairs(
    B: int,
    x_train: Sequence[int],
    residual_map: Dict[int, int],
) -> Tuple[List[float], List[int]]:
    """Algorithm-3 influence estimator using only observed neighbor pairs."""
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


def select_random_k(B: int, K: int, rng: random.Random) -> List[int]:
    k = min(K, B)
    return sorted(rng.sample(range(B), k))


def projected_residual_partial_tt(
    x_train: Sequence[int],
    r_train: Sequence[int],
    J: Sequence[int],
    rng: random.Random,
) -> List[str]:
    """Build projected empirical majority table on J; unseen patterns are '-' ."""
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


def train_multistage_residual(
    B: int,
    K: int,
    tau: float,
    stages_m: int,
    x_train: Sequence[int],
    y_train: Sequence[int],
    seed: int,
    selection_mode: str,
) -> Tuple[List[Stage], float, List[int]]:
    """
    Train a multi-stage XOR residual model.

    selection_mode:
      - "influence": choose top-K bits by estimated residual influence.
      - "random": choose K bits uniformly at random from [B] at each stage.

    Returns:
      stages, cumulative_training_time_seconds, selected_sizes_per_stage.

    The training time excludes test evaluation.
    """
    if selection_mode not in {"influence", "random"}:
        raise ValueError("selection_mode must be 'influence' or 'random'.")

    rng = random.Random(seed)
    stages: List[Stage] = []
    selected_sizes: List[int] = []
    cumulative_train_time = 0.0

    for _t in range(1, stages_m + 1):
        stage_start = time.perf_counter()

        r_train: List[int] = []
        residual_map: Dict[int, int] = {}
        for x, y in zip(x_train, y_train):
            r = int(y) ^ predict_stages(stages, x)
            r_train.append(r)
            residual_map[x] = r

        if selection_mode == "influence":
            infl, _pair_counts = influences_from_observed_pairs(B, x_train, residual_map)
            J = select_topk_above_tau(infl, K=K, tau=tau, rng=rng)
        else:
            J = select_random_k(B, K=K, rng=rng)

        tt_proj = projected_residual_partial_tt(x_train, r_train, J, rng=rng)
        model = espresso_learn_tt(tt_proj, n_vars=len(J))
        stages.append(Stage(bits=list(J), model=model))
        selected_sizes.append(len(J))

        cumulative_train_time += time.perf_counter() - stage_start

    return stages, cumulative_train_time, selected_sizes


# -----------------------------------------------------------------------------
# Multi-seed experiment and reporting
# -----------------------------------------------------------------------------

def mean_std(vals: Iterable[float]) -> Tuple[float, float]:
    vals = [float(v) for v in vals if float(v) == float(v)]  # drop NaN
    if not vals:
        return float("nan"), float("nan")
    if len(vals) == 1:
        return vals[0], 0.0
    return statistics.mean(vals), statistics.stdev(vals)  # sample std over seeds


def fmt_pm(mean: float, std: float, digits: int = 3) -> str:
    if mean != mean:
        return "NA"
    return f"{mean:.{digits}f} +- {std:.{digits}f}"


def run_experiment(args: argparse.Namespace) -> Path:
    require_pyeda()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    per_seed_rows: List[Dict[str, object]] = []

    for s in range(args.num_seeds):
        seed = args.base_seed + s
        data_rng = random.Random(seed)
        f = make_random_s_junta(args.B, args.S, data_rng)
        x_train, y_train = sample_dataset(f, args.B, args.train_size, data_rng)

        if args.full_cube_test:
            x_test, y_test = full_cube_dataset(f, args.B)
            test_size_used = 1 << args.B
        else:
            x_test, y_test = sample_dataset(f, args.B, args.test_size, data_rng)
            test_size_used = args.test_size

        print(f"[seed {seed}] junta_bits={f.junta_bits}")

        # Use different internal RNG streams for the two training procedures,
        # while keeping the same target function, train set, and test set.
        stages_inf, time_inf, sizes_inf = train_multistage_residual(
            B=args.B,
            K=args.K,
            tau=args.tau,
            stages_m=args.stages,
            x_train=x_train,
            y_train=y_train,
            seed=10_000_000 + seed,
            selection_mode="influence",
        )
        acc_inf = accuracy_from_predictor(lambda z: predict_stages(stages_inf, z), x_test, y_test)

        stages_rand, time_rand, sizes_rand = train_multistage_residual(
            B=args.B,
            K=args.K,
            tau=args.tau,
            stages_m=args.stages,
            x_train=x_train,
            y_train=y_train,
            seed=20_000_000 + seed,
            selection_mode="random",
        )
        acc_rand = accuracy_from_predictor(lambda z: predict_stages(stages_rand, z), x_test, y_test)

        print(f"  Influence top-K: acc={acc_inf:.4f}, train_time={time_inf:.2f}s")
        print(f"  Random K       : acc={acc_rand:.4f}, train_time={time_rand:.2f}s")

        per_seed_rows.append({
            "seed": seed,
            "B": args.B,
            "S": args.S,
            "K": args.K,
            "tau": args.tau,
            "train_size": args.train_size,
            "test_size": test_size_used,
            "stages": args.stages,
            "junta_bits": " ".join(map(str, f.junta_bits)),
            "influence_final_test_acc": acc_inf,
            "influence_train_time_s": time_inf,
            "influence_mean_selected_size": statistics.mean(sizes_inf) if sizes_inf else float("nan"),
            "random_final_test_acc": acc_rand,
            "random_train_time_s": time_rand,
            "random_mean_selected_size": statistics.mean(sizes_rand) if sizes_rand else float("nan"),
        })

    summary_rows: List[Dict[str, object]] = []
    methods = [
        ("Algorithm 3 (influence top-K)", "influence"),
        ("Algorithm 3 (random K ablation)", "random"),
    ]
    for method_name, prefix in methods:
        acc_mu, acc_sd = mean_std(row[f"{prefix}_final_test_acc"] for row in per_seed_rows)
        t_mu, t_sd = mean_std(row[f"{prefix}_train_time_s"] for row in per_seed_rows)
        k_mu, k_sd = mean_std(row[f"{prefix}_mean_selected_size"] for row in per_seed_rows)
        summary_rows.append({
            "method": method_name,
            "B": args.B,
            "S-junta": args.S,
            "K": args.K,
            "tau": args.tau,
            "train_size": args.train_size,
            "test_size": per_seed_rows[0]["test_size"] if per_seed_rows else args.test_size,
            "stages": args.stages,
            "final_test_accuracy_mean": acc_mu,
            "final_test_accuracy_std": acc_sd,
            "final_test_accuracy_mean_pm_std": fmt_pm(acc_mu, acc_sd),
            "train_runtime_s_mean": t_mu,
            "train_runtime_s_std": t_sd,
            "train_runtime_s_mean_pm_std": fmt_pm(t_mu, t_sd, digits=2),
            "mean_selected_size_mean": k_mu,
            "mean_selected_size_std": k_sd,
        })

    summary_path = outdir / "alg3_influence_vs_randomK_summary.csv"
    with summary_path.open("w", newline="") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\n=== Summary: mean +- sample std over seeds ===")
    header = f"{'Method':36s} | {'Final test acc':17s} | {'Train runtime [s]':20s}"
    print(header)
    print("-" * len(header))
    for r in summary_rows:
        print(
            f"{str(r['method']):36s} | "
            f"{r['final_test_accuracy_mean_pm_std']:17s} | "
            f"{r['train_runtime_s_mean_pm_std']:20s}"
        )

    print(f"\nSaved summary table to: {summary_path}")
    return summary_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--B", type=int, default=12, help="ambient input dimension")
    p.add_argument("--S", type=int, default=6, help="number of relevant variables in random S-junta")
    p.add_argument("--train-size", type=int, default=1000)
    p.add_argument("--test-size", type=int, default=2*12, help="sampled test size, ignored if --full-cube-test is used")
    p.add_argument("--full-cube-test", action="store_true", help="evaluate on the whole Boolean cube")
    p.add_argument("--K", type=int, default=4, help="number of variables selected per residual stage")
    p.add_argument("--tau", type=float, default=0.0, help="influence threshold for the influence top-K method")
    p.add_argument("--stages", type=int, default=20, help="number of residual stages")
    p.add_argument("--num-seeds", type=int, default=20)
    p.add_argument("--base-seed", type=int, default=0)
    p.add_argument("--outdir", type=str, default="results_alg3_influence_vs_random")
    return p.parse_args()


if __name__ == "__main__":
    run_experiment(parse_args())
