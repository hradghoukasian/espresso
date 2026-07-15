# #
# """
#
# Experiment: compare Algorithm 3 (sample-based multi-stage influence residual
# learning on a partial truth table) with flat Espresso on the ambient B-dimensional
# partial truth table.
# """
#
# from __future__ import annotations
#
# import argparse
# import csv
# import math
# import random
# import statistics
# import time
# from dataclasses import dataclass
# from pathlib import Path
# from typing import Dict, Iterable, List, Optional, Sequence, Tuple
#
# try:
#     from pyeda.inter import exprvars, truthtable
#     from pyeda.boolalg.minimization import espresso_tts
# except Exception as exc:  # pragma: no cover
#     exprvars = None
#     truthtable = None
#     espresso_tts = None
#     PYEDA_IMPORT_ERROR = exc
# else:
#     PYEDA_IMPORT_ERROR = None
#
#
# # -----------------------------------------------------------------------------
# # Basic utilities
# # -----------------------------------------------------------------------------
#
# def require_pyeda() -> None:
#     if PYEDA_IMPORT_ERROR is not None:
#         raise RuntimeError(
#             "PyEDA is required for the actual Espresso calls. Install it with:\n"
#             "    pip install pyeda\n"
#             f"Original import error: {PYEDA_IMPORT_ERROR!r}"
#         )
#
#
# def proj_index(x: int, cols: Sequence[int]) -> int:
#     """Project integer-coded bitstring x to the coordinates cols, packed as an int."""
#     u = 0
#     for j, bit in enumerate(cols):
#         if (x >> bit) & 1:
#             u |= 1 << j
#     return u
#
#
# @dataclass(frozen=True)
# class SJunta:
#     B: int
#     junta_bits: List[int]
#     junta_tt: List[int]
#
#     def __call__(self, x: int) -> int:
#         return self.junta_tt[proj_index(x, self.junta_bits)]
#
#
# def make_random_s_junta(B: int, S: int, rng: random.Random) -> SJunta:
#     if not (0 <= S <= B):
#         raise ValueError("Need 0 <= S <= B.")
#     junta_bits = sorted(rng.sample(range(B), S))
#     junta_tt = [rng.getrandbits(1) for _ in range(1 << S)]
#     return SJunta(B=B, junta_bits=junta_bits, junta_tt=junta_tt)
#
#
# def sample_dataset(f, B: int, T: int, rng: random.Random) -> Tuple[List[int], List[int]]:
#     """Uniform sample without replacement from {0,1}^B."""
#     cube_size = 1 << B
#     if T > cube_size:
#         raise ValueError(f"T={T} exceeds cube size 2^B={cube_size}.")
#     xs = rng.sample(range(cube_size), T)
#     ys = [int(f(x)) for x in xs]
#     return xs, ys
#
#
# def accuracy_from_predictor(predict_fn, xs: Sequence[int], ys: Sequence[int]) -> float:
#     if not xs:
#         return 0.0
#     correct = 0
#     for x, y in zip(xs, ys):
#         correct += int(int(predict_fn(x)) == int(y))
#     return correct / len(xs)
#
#
# # -----------------------------------------------------------------------------
# # Espresso wrapper for 0/1/- truth tables
# # -----------------------------------------------------------------------------
#
# @dataclass
# class EspressoModel:
#     n_vars: int
#     expr: object
#     constant_value: Optional[int] = None
#
#     def predict_one(self, x: int) -> int:
#         if self.constant_value is not None:
#             return int(self.constant_value)
#         X = exprvars("x", self.n_vars)
#         assignment = {X[j]: ((x >> j) & 1) for j in range(self.n_vars)}
#         val = self.expr.restrict(assignment)
#         return 1 if val.is_one() else 0
#
#
# def espresso_learn_tt(tt: Sequence[str], n_vars: int) -> EspressoModel:
#     """Run Espresso on a truth table over {'0','1','-'} and return a predictor."""
#     require_pyeda()
#     if len(tt) != (1 << n_vars):
#         raise ValueError(f"len(tt)={len(tt)} but expected 2^{n_vars}.")
#     chars = list(tt)
#     values = set(chars)
#     if not values.issubset({"0", "1", "-", "x"}):
#         raise ValueError(f"Truth table has invalid characters: {values}.")
#
#     # Degenerate constants/all-DC. This avoids edge cases in PyEDA.
#     specified = {c for c in chars if c in {"0", "1"}}
#     if n_vars == 0:
#         return EspressoModel(n_vars=0, expr=None, constant_value=1 if chars[0] == "1" else 0)
#     if specified == set():
#         return EspressoModel(n_vars=n_vars, expr=None, constant_value=0)
#     if specified == {"0"}:
#         return EspressoModel(n_vars=n_vars, expr=None, constant_value=0)
#     if specified == {"1"}:
#         return EspressoModel(n_vars=n_vars, expr=None, constant_value=1)
#
#     X = exprvars("x", n_vars)
#     f_tt = truthtable(X, "".join(chars))
#     expr, = espresso_tts(f_tt)
#     return EspressoModel(n_vars=n_vars, expr=expr)
#
#
# # -----------------------------------------------------------------------------
# # Algorithm 3: multi-stage influence-based residual learning on partial TT
# # -----------------------------------------------------------------------------
#
# @dataclass
# class Stage:
#     bits: List[int]
#     model: EspressoModel
#
#     def predict_one(self, x: int) -> int:
#         if not self.bits:
#             return self.model.predict_one(0)
#         return self.model.predict_one(proj_index(x, self.bits))
#
#
# def predict_algorithm3(stages: Sequence[Stage], x: int) -> int:
#     y = 0
#     for st in stages:
#         y ^= st.predict_one(x)
#     return y
#
#
# def influences_from_observed_pairs(
#     B: int,
#     x_train: Sequence[int],
#     residual_map: Dict[int, int],
# ) -> Tuple[List[float], List[int]]:
#     """Algorithm-3 influence estimator using only observed neighbor pairs."""
#     xset = set(x_train)
#     influences = [0.0] * B
#     pair_counts = [0] * B
#     for i in range(B):
#         mask = 1 << i
#         mismatches = 0
#         count = 0
#         for x in x_train:
#             xn = x ^ mask
#             if xn in xset:
#                 count += 1
#                 mismatches += int(residual_map[x] != residual_map[xn])
#         pair_counts[i] = count
#         influences[i] = (mismatches / count) if count else 0.0
#     return influences, pair_counts
#
#
# def select_topk_above_tau(influences: Sequence[float], K: int, tau: float, rng: random.Random) -> List[int]:
#     candidates = [i for i, v in enumerate(influences) if v > tau]
#     rng.shuffle(candidates)          # random tie-breaking before sorting
#     candidates.sort(key=lambda i: influences[i], reverse=True)
#     return candidates[: min(K, len(candidates))]
#
#
# def projected_residual_partial_tt(
#     x_train: Sequence[int],
#     r_train: Sequence[int],
#     J: Sequence[int],
#     rng: random.Random,
# ) -> List[str]:
#     """Build projected empirical majority table on J; unseen patterns are '-'."""
#     k = len(J)
#     if k == 0:
#         ones = sum(r_train)
#         zeros = len(r_train) - ones
#         if ones > zeros:
#             return ["1"]
#         if zeros > ones:
#             return ["0"]
#         return [str(rng.getrandbits(1))]
#
#     size = 1 << k
#     zeros = [0] * size
#     ones = [0] * size
#     for x, r in zip(x_train, r_train):
#         u = proj_index(x, J)
#         if r:
#             ones[u] += 1
#         else:
#             zeros[u] += 1
#
#     out: List[str] = []
#     for u in range(size):
#         if zeros[u] == 0 and ones[u] == 0:
#             out.append("-")
#         elif ones[u] > zeros[u]:
#             out.append("1")
#         elif zeros[u] > ones[u]:
#             out.append("0")
#         else:
#             out.append(str(rng.getrandbits(1)))
#     return out
#
#
# def train_algorithm3(
#     B: int,
#     K: int,
#     tau: float,
#     stages_m: int,
#     x_train: Sequence[int],
#     y_train: Sequence[int],
#     x_test: Sequence[int],
#     y_test: Sequence[int],
#     seed: int,
#     stage_points: Sequence[int],
# ) -> Tuple[List[Stage], Dict[int, float], Dict[int, float], Dict[int, int]]:
#     """
#     Returns stages, test accuracy at requested stages, cumulative TRAINING time at
#     requested stages, and selected |J_t| at requested stages.
#
#     Important: test-set evaluation is deliberately NOT included in the reported
#     training time.  We accumulate only the time spent in the training operations
#     of each stage: residual computation on D_train, influence estimation on
#     observed training pairs, top-K variable selection, projected partial truth
#     table construction, and Espresso minimization on the projected table.
#     """
#     rng = random.Random(seed)
#     stages: List[Stage] = []
#     stage_set = set(stage_points)
#     acc_at: Dict[int, float] = {}
#     time_at: Dict[int, float] = {}
#     selected_sizes: Dict[int, int] = {}
#
#     cumulative_train_time = 0.0
#
#     for t in range(1, stages_m + 1):
#         stage_train_start = time.perf_counter()
#
#         r_train: List[int] = []
#         residual_map: Dict[int, int] = {}
#         for x, y in zip(x_train, y_train):
#             r = int(y) ^ predict_algorithm3(stages, x)
#             r_train.append(r)
#             residual_map[x] = r
#
#         infl, _pair_counts = influences_from_observed_pairs(B, x_train, residual_map)
#         J = select_topk_above_tau(infl, K=K, tau=tau, rng=rng)
#         tt_proj = projected_residual_partial_tt(x_train, r_train, J, rng=rng)
#         model = espresso_learn_tt(tt_proj, n_vars=len(J))
#         stages.append(Stage(bits=list(J), model=model))
#
#         cumulative_train_time += time.perf_counter() - stage_train_start
#
#         if t in stage_set:
#             # Accuracy is reported, but its cost is not included in training time.
#             acc = accuracy_from_predictor(lambda z: predict_algorithm3(stages, z), x_test, y_test)
#             acc_at[t] = acc
#             time_at[t] = cumulative_train_time
#             selected_sizes[t] = len(J)
#
#     return stages, acc_at, time_at, selected_sizes
#
#
# # -----------------------------------------------------------------------------
# # Flat ambient Espresso baseline
# # -----------------------------------------------------------------------------
#
# def build_ambient_partial_tt(B: int, x_train: Sequence[int], y_train: Sequence[int]) -> List[str]:
#     tt = ["-"] * (1 << B)
#     for x, y in zip(x_train, y_train):
#         tt[x] = "1" if int(y) else "0"
#     return tt
#
#
# def train_flat_espresso(
#     B: int,
#     x_train: Sequence[int],
#     y_train: Sequence[int],
#     x_test: Sequence[int],
#     y_test: Sequence[int],
# ) -> Tuple[float, float]:
#     """Return (test_accuracy, train_runtime_seconds)."""
#     start = time.perf_counter()
#     tt = build_ambient_partial_tt(B, x_train, y_train)
#     model = espresso_learn_tt(tt, n_vars=B)
#     train_time = time.perf_counter() - start
#     acc = accuracy_from_predictor(model.predict_one, x_test, y_test)
#     return acc, train_time
#
#
# # -----------------------------------------------------------------------------
# # Multi-seed experiment and reporting
# # -----------------------------------------------------------------------------
#
# def mean_std(vals: Iterable[float]) -> Tuple[float, float]:
#     vals = [v for v in vals if v == v]  # drop NaN
#     if not vals:
#         return float("nan"), float("nan")
#     if len(vals) == 1:
#         return vals[0], 0.0
#     return statistics.mean(vals), statistics.stdev(vals)  # sample std over seeds
#
#
# def fmt_pm(mean: float, std: float, digits: int = 3) -> str:
#     if mean != mean:
#         return "NA"
#     return f"{mean:.{digits}f} +- {std:.{digits}f}"
#
#
# def run_experiment(args: argparse.Namespace) -> Path:
#     require_pyeda()
#     outdir = Path(args.outdir)
#     outdir.mkdir(parents=True, exist_ok=True)
#
#     stage_points = sorted(set(args.stage_points))
#     if not stage_points:
#         stage_points = [args.stages]
#     if min(stage_points) < 1 or max(stage_points) > args.stages:
#         raise ValueError("All --stage-points must be between 1 and --stages.")
#
#     per_seed_rows: List[Dict[str, object]] = []
#
#     for s in range(args.num_seeds):
#         seed = args.base_seed + s
#         rng = random.Random(seed)
#         f = make_random_s_junta(args.B, args.S, rng)
#         x_train, y_train = sample_dataset(f, args.B, args.train_size, rng)
#         x_test, y_test = sample_dataset(f, args.B, args.test_size, rng)
#
#         print(f"[seed {seed}] junta_bits={f.junta_bits}")
#
#         _stages, alg_acc_at, alg_time_at, selected_sizes = train_algorithm3(
#             B=args.B,
#             K=args.K,
#             tau=args.tau,
#             stages_m=args.stages,
#             x_train=x_train,
#             y_train=y_train,
#             x_test=x_test,
#             y_test=y_test,
#             seed=seed,
#             stage_points=stage_points,
#         )
#
#         # Print Algorithm 3 results immediately, before starting flat Espresso.
#         alg_status_now = ", ".join(
#             f"A3@{sp}: acc={alg_acc_at.get(sp, float('nan')):.4f}, "
#             f"time={alg_time_at.get(sp, float('nan')):.2f}s"
#             for sp in stage_points
#         )
#         print(f"  {alg_status_now}", flush=True)
#
#         # Flat Espresso may be infeasible for large B because it constructs a length 2^B table.
#         flat_acc = float("nan")
#         flat_time = float("nan")
#         flat_status = "skipped"
#         print("  Starting flat Espresso...", flush=True)
#         if args.B <= args.max_flat_bits:
#             try:
#                 flat_acc, flat_time = train_flat_espresso(args.B, x_train, y_train, x_test, y_test)
#                 flat_status = "ok"
#             except Exception as exc:
#                 flat_status = f"failed: {type(exc).__name__}: {exc}"
#         else:
#             flat_status = f"skipped: B={args.B} > max_flat_bits={args.max_flat_bits}"
#
#         # Print flat Espresso result immediately after it finishes.
#         print(f"  Flat: status={flat_status}, acc={flat_acc:.4f}, time={flat_time:.2f}s", flush=True)
#
#         row: Dict[str, object] = {
#             "seed": seed,
#             "B": args.B,
#             "S": args.S,
#             "K": args.K,
#             "tau": args.tau,
#             "train_size": args.train_size,
#             "test_size": args.test_size,
#             "stages": args.stages,
#             "junta_bits": " ".join(map(str, f.junta_bits)),
#             "flat_status": flat_status,
#             "flat_test_acc": flat_acc,
#             "flat_train_time_s": flat_time,
#         }
#         for sp in stage_points:
#             row[f"alg3_test_acc_stage_{sp}"] = alg_acc_at.get(sp, float("nan"))
#             row[f"alg3_train_time_s_stage_{sp}"] = alg_time_at.get(sp, float("nan"))
#             row[f"alg3_selected_size_stage_{sp}"] = selected_sizes.get(sp, float("nan"))
#         per_seed_rows.append(row)
#
#
#     summary_rows: List[Dict[str, object]] = []
#     for sp in stage_points:
#         acc_mu, acc_sd = mean_std(row[f"alg3_test_acc_stage_{sp}"] for row in per_seed_rows)
#         t_mu, t_sd = mean_std(row[f"alg3_train_time_s_stage_{sp}"] for row in per_seed_rows)
#         summary_rows.append({
#             "method": f"Algorithm 3 (stage {sp})",
#             "B": args.B,
#             "S-junta": args.S,
#             "K": args.K,
#             "train_size": args.train_size,
#             "test_size": args.test_size,
#             "test_accuracy_mean": acc_mu,
#             "test_accuracy_std": acc_sd,
#             "test_accuracy_mean_pm_std": fmt_pm(acc_mu, acc_sd),
#             "train_runtime_s_mean": t_mu,
#             "train_runtime_s_std": t_sd,
#             "train_runtime_s_mean_pm_std": fmt_pm(t_mu, t_sd, digits=2),
#         })
#
#     flat_acc_mu, flat_acc_sd = mean_std(row["flat_test_acc"] for row in per_seed_rows)
#     flat_t_mu, flat_t_sd = mean_std(row["flat_train_time_s"] for row in per_seed_rows)
#     summary_rows.append({
#         "method": "Flat Espresso (ambient B partial TT)",
#         "B": args.B,
#         "S-junta": args.S,
#         "K": args.K,
#         "train_size": args.train_size,
#         "test_size": args.test_size,
#         "test_accuracy_mean": flat_acc_mu,
#         "test_accuracy_std": flat_acc_sd,
#         "test_accuracy_mean_pm_std": fmt_pm(flat_acc_mu, flat_acc_sd),
#         "train_runtime_s_mean": flat_t_mu,
#         "train_runtime_s_std": flat_t_sd,
#         "train_runtime_s_mean_pm_std": fmt_pm(flat_t_mu, flat_t_sd, digits=2),
#     })
#
#     summary_path = outdir / "summary_table.csv"
#     with summary_path.open("w", newline="") as fcsv:
#         writer = csv.DictWriter(fcsv, fieldnames=list(summary_rows[0].keys()))
#         writer.writeheader()
#         writer.writerows(summary_rows)
#
#     print("\n=== Summary: mean +- sample std over seeds ===")
#     header = f"{'Method':42s} | {'Test acc':17s} | {'Train runtime [s]':20s}"
#     print(header)
#     print("-" * len(header))
#     for r in summary_rows:
#         print(f"{str(r['method']):42s} | {r['test_accuracy_mean_pm_std']:17s} | {r['train_runtime_s_mean_pm_std']:20s}")
#
#     print(f"\nSaved summary table to: {summary_path}")
#     return summary_path
#
#
# def parse_args() -> argparse.Namespace:
#     p = argparse.ArgumentParser()
#     p.add_argument("--B", type=int, default=20, help="ambient input dimension")
#     p.add_argument("--S", type=int, default=12, help="number of relevant variables in random S-junta")
#     p.add_argument("--train-size", type=int, default=2**)
#     p.add_argument("--test-size", type=int, default=2**17)
#     p.add_argument("--K", type=int, default=10, help="top-K selected variables per residual stage")
#     p.add_argument("--tau", type=float, default=0.0, help="influence threshold")
#     p.add_argument("--stages", type=int, default=20, help="number of residual stages")
#     p.add_argument("--stage-points", type=int, nargs="+", default=[1, 5,20], help="stages at which to report Algorithm 3 accuracy/runtime")
#     p.add_argument("--num-seeds", type=int, default=20)
#     p.add_argument("--base-seed", type=int, default=0)
#     p.add_argument("--max-flat-bits", type=int, default=25, help="skip flat ambient Espresso if B is larger than this")
#     p.add_argument("--outdir", type=str, default="results_alg3_vs_flat")
#     return p.parse_args()
#
#
# if __name__ == "__main__":
#     run_experiment(parse_args())


## Cached Implementation

"""
Experiment: compare Algorithm 3 (sample-based multi-stage influence-residual
learning on a partial truth table) with flat Espresso on the ambient
B-dimensional partial truth table.

Optimization:
- Algorithm 3 caches each learned projected stage as a complete truth table.
- Later residual computations use fast list lookups instead of repeatedly
  evaluating PyEDA expressions.
- The one-time caching cost is included in Algorithm 3's reported training time.
- Test-set evaluation time is excluded from all reported training times.
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
    from pyeda.boolalg.minimization import espresso_tts
    from pyeda.inter import exprvars, truthtable
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


# -----------------------------------------------------------------------------
# Basic Boolean utilities
# -----------------------------------------------------------------------------

def proj_index(x: int, cols: Sequence[int]) -> int:
    """Project integer-coded bitstring x onto cols and pack the result as an int."""
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
    if not 0 <= S <= B:
        raise ValueError("Need 0 <= S <= B.")

    junta_bits = sorted(rng.sample(range(B), S))
    junta_tt = [rng.getrandbits(1) for _ in range(1 << S)]
    return SJunta(B=B, junta_bits=junta_bits, junta_tt=junta_tt)


def sample_dataset(
    f,
    B: int,
    T: int,
    rng: random.Random,
) -> Tuple[List[int], List[int]]:
    """Uniformly sample without replacement from {0,1}^B."""
    cube_size = 1 << B
    if T > cube_size:
        raise ValueError(f"T={T} exceeds cube size 2^B={cube_size}.")

    xs = rng.sample(range(cube_size), T)
    ys = [int(f(x)) for x in xs]
    return xs, ys


def accuracy_from_predictor(
    predict_fn,
    xs: Sequence[int],
    ys: Sequence[int],
) -> float:
    if not xs:
        return 0.0

    correct = sum(
        int(int(predict_fn(int(x))) == int(y))
        for x, y in zip(xs, ys)
    )
    return correct / len(xs)


# -----------------------------------------------------------------------------
# Espresso wrapper
# -----------------------------------------------------------------------------

@dataclass
class EspressoModel:
    """
    Espresso model supporting either:
      1) a cached complete truth table, or
      2) symbolic PyEDA evaluation.

    Algorithm 3 uses cached truth tables.
    Flat Espresso keeps the symbolic representation to avoid caching 2^B values.
    """

    n_vars: int
    expr: object = None
    constant_value: Optional[int] = None
    tt_full: Optional[List[int]] = None
    variables: object = None

    def predict_one(self, x: int) -> int:
        if self.tt_full is not None:
            return int(self.tt_full[x])

        if self.constant_value is not None:
            return int(self.constant_value)

        if self.expr is None or self.variables is None:
            raise RuntimeError("Espresso model has no valid prediction representation.")

        assignment = {
            self.variables[j]: ((x >> j) & 1)
            for j in range(self.n_vars)
        }
        value = self.expr.restrict(assignment)
        return 1 if value.is_one() else 0


def espresso_learn_tt(
    tt: Sequence[str],
    n_vars: int,
    *,
    cache_full_table: bool,
) -> EspressoModel:
    """
    Run Espresso on a truth table over {'0', '1', '-', 'x'}.

    When cache_full_table=True, evaluate the learned expression once on all
    2^n_vars Boolean inputs and store the outputs for fast future prediction.
    """
    require_pyeda()

    expected_size = 1 << n_vars
    if len(tt) != expected_size:
        raise ValueError(
            f"len(tt)={len(tt)}, but expected 2^{n_vars}={expected_size}."
        )

    chars = list(tt)
    values = set(chars)
    if not values.issubset({"0", "1", "-", "x"}):
        raise ValueError(f"Truth table contains invalid symbols: {values}.")

    specified = {c for c in chars if c in {"0", "1"}}

    # Degenerate cases
    if n_vars == 0:
        value = 1 if chars[0] == "1" else 0
        return EspressoModel(
            n_vars=0,
            constant_value=value,
            tt_full=[value] if cache_full_table else None,
        )

    if not specified or specified == {"0"}:
        return EspressoModel(
            n_vars=n_vars,
            constant_value=0,
            tt_full=[0] * expected_size if cache_full_table else None,
        )

    if specified == {"1"}:
        return EspressoModel(
            n_vars=n_vars,
            constant_value=1,
            tt_full=[1] * expected_size if cache_full_table else None,
        )

    X = exprvars("x", n_vars)
    f_tt = truthtable(X, "".join(chars))
    expr, = espresso_tts(f_tt)

    if not cache_full_table:
        return EspressoModel(
            n_vars=n_vars,
            expr=expr,
            variables=X,
        )

    # One-time symbolic evaluation; later predictions are list lookups.
    full_predictions: List[int] = []
    for u in range(expected_size):
        assignment = {X[j]: ((u >> j) & 1) for j in range(n_vars)}
        value = expr.restrict(assignment)
        full_predictions.append(1 if value.is_one() else 0)

    return EspressoModel(
        n_vars=n_vars,
        expr=expr,
        tt_full=full_predictions,
        variables=X,
    )


# -----------------------------------------------------------------------------
# Algorithm 3
# -----------------------------------------------------------------------------

@dataclass
class Stage:
    bits: List[int]
    tt_full: List[int]

    def predict_one(self, x: int) -> int:
        return int(self.tt_full[proj_index(x, self.bits)])


def predict_algorithm3(stages: Sequence[Stage], x: int) -> int:
    prediction = 0
    for stage in stages:
        prediction ^= stage.predict_one(x)
    return prediction


def influences_from_observed_pairs(
    B: int,
    x_train: Sequence[int],
    residual_map: Dict[int, int],
) -> Tuple[List[float], List[int]]:
    """Estimate influences using only observed Hamming-neighbor pairs."""
    xset = set(x_train)
    influences = [0.0] * B
    pair_counts = [0] * B

    for i in range(B):
        mask = 1 << i
        mismatches = 0
        count = 0

        for x in x_train:
            neighbor = x ^ mask
            if neighbor in xset:
                count += 1
                mismatches += int(
                    residual_map[x] != residual_map[neighbor]
                )

        pair_counts[i] = count
        influences[i] = mismatches / count if count else 0.0

    return influences, pair_counts


def select_topk_above_tau(
    influences: Sequence[float],
    K: int,
    tau: float,
    rng: random.Random,
) -> List[int]:
    candidates = [
        i for i, influence in enumerate(influences)
        if influence > tau
    ]

    # Randomize ties before stable sorting.
    rng.shuffle(candidates)
    candidates.sort(
        key=lambda i: influences[i],
        reverse=True,
    )
    return candidates[: min(K, len(candidates))]


def projected_residual_partial_tt(
    x_train: Sequence[int],
    r_train: Sequence[int],
    selected_bits: Sequence[int],
    rng: random.Random,
) -> List[str]:
    """
    Build the projected empirical-majority truth table.
    Unobserved projected inputs are don't-cares '-'.
    """
    k = len(selected_bits)

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

    for x, residual in zip(x_train, r_train):
        u = proj_index(x, selected_bits)
        if residual:
            ones[u] += 1
        else:
            zeros[u] += 1

    output: List[str] = []
    for u in range(size):
        if zeros[u] == 0 and ones[u] == 0:
            output.append("-")
        elif ones[u] > zeros[u]:
            output.append("1")
        elif zeros[u] > ones[u]:
            output.append("0")
        else:
            output.append(str(rng.getrandbits(1)))

    return output


def train_algorithm3(
    B: int,
    K: int,
    tau: float,
    stages_m: int,
    x_train: Sequence[int],
    y_train: Sequence[int],
    x_test: Sequence[int],
    y_test: Sequence[int],
    seed: int,
    stage_points: Sequence[int],
) -> Tuple[
    List[Stage],
    Dict[int, float],
    Dict[int, float],
    Dict[int, int],
]:
    """
    Train Algorithm 3.

    Returns:
      - learned stages,
      - test accuracy at requested stage points,
      - cumulative training time at requested stage points,
      - selected-set size at requested stage points.

    The reported training time includes:
      - residual computation on D_train,
      - influence estimation,
      - top-K selection,
      - projected truth-table construction,
      - Espresso minimization,
      - one-time caching of each projected stage truth table.

    Test-set evaluation is excluded.
    """
    rng = random.Random(seed)
    stages: List[Stage] = []

    requested_stages = set(stage_points)
    accuracy_at: Dict[int, float] = {}
    time_at: Dict[int, float] = {}
    selected_sizes: Dict[int, int] = {}

    cumulative_train_time = 0.0

    for t in range(1, stages_m + 1):
        stage_start = time.perf_counter()

        residuals: List[int] = []
        residual_map: Dict[int, int] = {}

        for x, y in zip(x_train, y_train):
            residual = int(y) ^ predict_algorithm3(stages, x)
            residuals.append(residual)
            residual_map[x] = residual

        influences, _pair_counts = influences_from_observed_pairs(
            B=B,
            x_train=x_train,
            residual_map=residual_map,
        )

        selected_bits = select_topk_above_tau(
            influences=influences,
            K=K,
            tau=tau,
            rng=rng,
        )

        projected_tt = projected_residual_partial_tt(
            x_train=x_train,
            r_train=residuals,
            selected_bits=selected_bits,
            rng=rng,
        )

        model = espresso_learn_tt(
            projected_tt,
            n_vars=len(selected_bits),
            cache_full_table=True,
        )

        if model.tt_full is None:
            raise RuntimeError("Expected a cached truth table for Algorithm 3.")

        stages.append(
            Stage(
                bits=list(selected_bits),
                tt_full=list(model.tt_full),
            )
        )

        cumulative_train_time += time.perf_counter() - stage_start

        if t in requested_stages:
            # Evaluation occurs after the training timer has stopped.
            accuracy_at[t] = accuracy_from_predictor(
                lambda z: predict_algorithm3(stages, z),
                x_test,
                y_test,
            )
            time_at[t] = cumulative_train_time
            selected_sizes[t] = len(selected_bits)

    return stages, accuracy_at, time_at, selected_sizes


# -----------------------------------------------------------------------------
# Flat ambient Espresso baseline
# -----------------------------------------------------------------------------

def build_ambient_partial_tt(
    B: int,
    x_train: Sequence[int],
    y_train: Sequence[int],
) -> List[str]:
    table = ["-"] * (1 << B)
    for x, y in zip(x_train, y_train):
        table[x] = "1" if int(y) else "0"
    return table


def train_flat_espresso(
    B: int,
    x_train: Sequence[int],
    y_train: Sequence[int],
    x_test: Sequence[int],
    y_test: Sequence[int],
) -> Tuple[float, float]:
    """
    Return (test_accuracy, training_runtime_seconds).

    Flat Espresso does not cache all 2^B predictions because that can be large.
    Test evaluation remains outside the reported training time.
    """
    start = time.perf_counter()

    ambient_tt = build_ambient_partial_tt(
        B=B,
        x_train=x_train,
        y_train=y_train,
    )
    model = espresso_learn_tt(
        ambient_tt,
        n_vars=B,
        cache_full_table=False,
    )

    training_time = time.perf_counter() - start

    accuracy = accuracy_from_predictor(
        model.predict_one,
        x_test,
        y_test,
    )
    return accuracy, training_time


# -----------------------------------------------------------------------------
# Reporting helpers
# -----------------------------------------------------------------------------

def mean_std(values: Iterable[float]) -> Tuple[float, float]:
    clean_values = [
        float(value)
        for value in values
        if float(value) == float(value)
    ]

    if not clean_values:
        return float("nan"), float("nan")

    if len(clean_values) == 1:
        return clean_values[0], 0.0

    return (
        statistics.mean(clean_values),
        statistics.stdev(clean_values),
    )


def fmt_pm(mean: float, std: float, digits: int = 3) -> str:
    if mean != mean:
        return "NA"
    return f"{mean:.{digits}f} +- {std:.{digits}f}"


# -----------------------------------------------------------------------------
# Main experiment
# -----------------------------------------------------------------------------

def run_experiment(args: argparse.Namespace) -> Path:
    require_pyeda()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    stage_points = sorted(set(args.stage_points))
    if not stage_points:
        stage_points = [args.stages]

    if min(stage_points) < 1 or max(stage_points) > args.stages:
        raise ValueError(
            "All --stage-points must lie between 1 and --stages."
        )

    per_seed_rows: List[Dict[str, object]] = []

    for seed_offset in range(args.num_seeds):
        seed = args.base_seed + seed_offset
        rng = random.Random(seed)

        target = make_random_s_junta(
            B=args.B,
            S=args.S,
            rng=rng,
        )

        x_train, y_train = sample_dataset(
            target,
            B=args.B,
            T=args.train_size,
            rng=rng,
        )
        x_test, y_test = sample_dataset(
            target,
            B=args.B,
            T=args.test_size,
            rng=rng,
        )

        print(f"[seed {seed}] junta_bits={target.junta_bits}")

        (
            _stages,
            algorithm3_accuracy,
            algorithm3_time,
            selected_sizes,
        ) = train_algorithm3(
            B=args.B,
            K=args.K,
            tau=args.tau,
            stages_m=args.stages,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            seed=seed,
            stage_points=stage_points,
        )

        algorithm3_status = ", ".join(
            (
                f"A3@{stage}: "
                f"acc={algorithm3_accuracy.get(stage, float('nan')):.4f}, "
                f"time={algorithm3_time.get(stage, float('nan')):.2f}s"
            )
            for stage in stage_points
        )
        print(f"  {algorithm3_status}", flush=True)

        flat_accuracy = float("nan")
        flat_time = float("nan")
        flat_status = "skipped"

        print("  Starting flat Espresso...", flush=True)

        if args.B <= args.max_flat_bits:
            try:
                flat_accuracy, flat_time = train_flat_espresso(
                    B=args.B,
                    x_train=x_train,
                    y_train=y_train,
                    x_test=x_test,
                    y_test=y_test,
                )
                flat_status = "ok"
            except Exception as exc:
                flat_status = (
                    f"failed: {type(exc).__name__}: {exc}"
                )
        else:
            flat_status = (
                f"skipped: B={args.B} > "
                f"max_flat_bits={args.max_flat_bits}"
            )

        print(
            f"  Flat: status={flat_status}, "
            f"acc={flat_accuracy:.4f}, "
            f"time={flat_time:.2f}s",
            flush=True,
        )

        row: Dict[str, object] = {
            "seed": seed,
            "B": args.B,
            "S": args.S,
            "K": args.K,
            "tau": args.tau,
            "train_size": args.train_size,
            "test_size": args.test_size,
            "stages": args.stages,
            "junta_bits": " ".join(map(str, target.junta_bits)),
            "flat_status": flat_status,
            "flat_test_acc": flat_accuracy,
            "flat_train_time_s": flat_time,
        }

        for stage in stage_points:
            row[f"alg3_test_acc_stage_{stage}"] = (
                algorithm3_accuracy.get(stage, float("nan"))
            )
            row[f"alg3_train_time_s_stage_{stage}"] = (
                algorithm3_time.get(stage, float("nan"))
            )
            row[f"alg3_selected_size_stage_{stage}"] = (
                selected_sizes.get(stage, float("nan"))
            )

        per_seed_rows.append(row)

    summary_rows: List[Dict[str, object]] = []

    for stage in stage_points:
        accuracy_mean, accuracy_std = mean_std(
            row[f"alg3_test_acc_stage_{stage}"]
            for row in per_seed_rows
        )
        time_mean, time_std = mean_std(
            row[f"alg3_train_time_s_stage_{stage}"]
            for row in per_seed_rows
        )

        summary_rows.append(
            {
                "method": f"Algorithm 3 (stage {stage})",
                "B": args.B,
                "S-junta": args.S,
                "K": args.K,
                "train_size": args.train_size,
                "test_size": args.test_size,
                "test_accuracy_mean": accuracy_mean,
                "test_accuracy_std": accuracy_std,
                "test_accuracy_mean_pm_std": fmt_pm(
                    accuracy_mean,
                    accuracy_std,
                ),
                "train_runtime_s_mean": time_mean,
                "train_runtime_s_std": time_std,
                "train_runtime_s_mean_pm_std": fmt_pm(
                    time_mean,
                    time_std,
                    digits=2,
                ),
            }
        )

    flat_accuracy_mean, flat_accuracy_std = mean_std(
        row["flat_test_acc"]
        for row in per_seed_rows
    )
    flat_time_mean, flat_time_std = mean_std(
        row["flat_train_time_s"]
        for row in per_seed_rows
    )

    summary_rows.append(
        {
            "method": "Flat Espresso (ambient B partial TT)",
            "B": args.B,
            "S-junta": args.S,
            "K": args.K,
            "train_size": args.train_size,
            "test_size": args.test_size,
            "test_accuracy_mean": flat_accuracy_mean,
            "test_accuracy_std": flat_accuracy_std,
            "test_accuracy_mean_pm_std": fmt_pm(
                flat_accuracy_mean,
                flat_accuracy_std,
            ),
            "train_runtime_s_mean": flat_time_mean,
            "train_runtime_s_std": flat_time_std,
            "train_runtime_s_mean_pm_std": fmt_pm(
                flat_time_mean,
                flat_time_std,
                digits=2,
            ),
        }
    )

    summary_path = outdir / "summary_table.csv"
    with summary_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=list(summary_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    if args.save_per_seed:
        per_seed_path = outdir / "per_seed_results.csv"
        with per_seed_path.open("w", newline="") as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=list(per_seed_rows[0].keys()),
            )
            writer.writeheader()
            writer.writerows(per_seed_rows)

        print(f"Saved per-seed results to: {per_seed_path}")

    print("\n=== Summary: mean +- sample std over seeds ===")
    header = (
        f"{'Method':42s} | "
        f"{'Test acc':17s} | "
        f"{'Train runtime [s]':20s}"
    )
    print(header)
    print("-" * len(header))

    for row in summary_rows:
        print(
            f"{str(row['method']):42s} | "
            f"{row['test_accuracy_mean_pm_std']:17s} | "
            f"{row['train_runtime_s_mean_pm_std']:20s}"
        )

    print(f"\nSaved summary table to: {summary_path}")
    return summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--B",
        type=int,
        default=20,
        help="ambient input dimension",
    )
    parser.add_argument(
        "--S",
        type=int,
        default=8,
        help="number of relevant variables in the random S-junta",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=2**20,
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=2**17,
    )
    parser.add_argument(
        "--K",
        type=int,
        default=6,
        help="maximum selected variables per residual stage",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.0,
        help="influence threshold",
    )
    parser.add_argument(
        "--stages",
        type=int,
        default=20,
        help="number of residual stages",
    )
    parser.add_argument(
        "--stage-points",
        type=int,
        nargs="+",
        default=[1, 5, 20],
        help="stages at which Algorithm 3 accuracy/runtime are reported",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--max-flat-bits",
        type=int,
        default=0,
        help="skip flat Espresso when B exceeds this value",
    )
    parser.add_argument(
        "--save-per-seed",
        action="store_true",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="results_alg3_vs_flat_cached",
    )

    return parser.parse_args()


if __name__ == "__main__":
    run_experiment(parse_args())