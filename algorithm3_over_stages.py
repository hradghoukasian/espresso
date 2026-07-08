"""
algorithm3_accuracy_over_stages.py

Experiment: run Algorithm 3 (sample-based multi-stage influence residual learning
on a partial truth table) and report test accuracy over residual stages.


Ablation plot is generated: test accuracy vs number of residual stages.
Training runtime is cumulative through each reported stage and excludes test
accuracy evaluation time.

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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


def make_full_cube_dataset(f, B: int) -> Tuple[List[int], List[int]]:
    """Use the entire Boolean cube as the test set."""
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
# Algorithm 3
# -----------------------------------------------------------------------------

@dataclass
class Stage:
    bits: List[int]
    model: EspressoModel

    def predict_one(self, x: int) -> int:
        if not self.bits:
            return self.model.predict_one(0)
        return self.model.predict_one(proj_index(x, self.bits))


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
    rng.shuffle(candidates)          # random tie-breaking before sorting
    candidates.sort(key=lambda i: influences[i], reverse=True)
    return candidates[: min(K, len(candidates))]


def projected_residual_partial_tt(
    x_train: Sequence[int],
    r_train: Sequence[int],
    J: Sequence[int],
    rng: random.Random,
) -> List[str]:
    """Build projected empirical majority table on J; unseen patterns are '-'."""
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
    x_test: Sequence[int],
    y_test: Sequence[int],
    seed: int,
    stage_points: Sequence[int],
) -> Tuple[List[Stage], Dict[int, float], Dict[int, float], Dict[int, int]]:
    """
    Returns stages, test accuracy at requested stages, cumulative training time at
    requested stages, and selected |J_t| at requested stages.

    Timing note: training time excludes test accuracy evaluation time.
    """
    rng = random.Random(seed)
    stages: List[Stage] = []
    stage_set = set(stage_points)
    acc_at: Dict[int, float] = {}
    time_at: Dict[int, float] = {}
    selected_sizes: Dict[int, int] = {}
    cumulative_train_time = 0.0

    for t in range(1, stages_m + 1):
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
        stages.append(Stage(bits=list(J), model=model))

        cumulative_train_time += time.perf_counter() - stage_start

        if t in stage_set:
            acc = accuracy_from_predictor(lambda z: predict_algorithm3(stages, z), x_test, y_test)
            acc_at[t] = acc
            time_at[t] = cumulative_train_time
            selected_sizes[t] = len(J)

    return stages, acc_at, time_at, selected_sizes


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------

def mean_std(vals: Iterable[float]) -> Tuple[float, float]:
    vals = [v for v in vals if v == v]
    if not vals:
        return float("nan"), float("nan")
    if len(vals) == 1:
        return vals[0], 0.0
    return statistics.mean(vals), statistics.stdev(vals)


def fmt_pm(mean: float, std: float, digits: int = 3) -> str:
    if mean != mean:
        return "NA"
    return f"{mean:.{digits}f} +- {std:.{digits}f}"


def run_experiment(args: argparse.Namespace) -> Path:
    require_pyeda()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Stages shown in the summary CSV/table.
    stage_points = sorted(set(args.stage_points))
    if not stage_points:
        stage_points = [args.stages]
    if min(stage_points) < 1 or max(stage_points) > args.stages:
        raise ValueError("All --stage-points must be between 1 and --stages.")

    # Stages used for the ablation plot.
    # By default this is every stage 1,2,...,m, but the summary table can
    # still report only selected stages such as 1,5,20.
    if args.make_stage_plot:
        if args.plot_every_stage:
            plot_stage_points = list(range(1, args.stages + 1))
        else:
            plot_stage_points = sorted(set(args.plot_stage_points or args.stage_points))
    else:
        plot_stage_points = []

    train_eval_stage_points = sorted(set(stage_points).union(plot_stage_points))

    per_seed_rows: List[Dict[str, object]] = []

    for s in range(args.num_seeds):
        seed = args.base_seed + s
        rng = random.Random(seed)
        f = make_random_s_junta(args.B, args.S, rng)
        x_train, y_train = sample_dataset(f, args.B, args.train_size, rng)
        if args.full_cube_test:
            x_test, y_test = make_full_cube_dataset(f, args.B)
        else:
            x_test, y_test = sample_dataset(f, args.B, args.test_size, rng)

        print(f"[seed {seed}] junta_bits={f.junta_bits}")

        _stages, alg_acc_at, alg_time_at, selected_sizes = train_algorithm3(
            B=args.B,
            K=args.K,
            tau=args.tau,
            stages_m=args.stages,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            seed=seed,
            stage_points=train_eval_stage_points,
        )

        row: Dict[str, object] = {
            "seed": seed,
            "B": args.B,
            "S": args.S,
            "K": args.K,
            "tau": args.tau,
            "train_size": args.train_size,
            "test_size": len(x_test),
            "stages": args.stages,
            "junta_bits": " ".join(map(str, f.junta_bits)),
        }
        for sp in train_eval_stage_points:
            row[f"alg3_test_acc_stage_{sp}"] = alg_acc_at.get(sp, float("nan"))
            row[f"alg3_train_time_s_stage_{sp}"] = alg_time_at.get(sp, float("nan"))
            row[f"alg3_selected_size_stage_{sp}"] = selected_sizes.get(sp, float("nan"))
        per_seed_rows.append(row)

        alg_status = ", ".join(
            f"A3@{sp}: acc={row[f'alg3_test_acc_stage_{sp}']:.4f}, "
            f"train_time={row[f'alg3_train_time_s_stage_{sp}']:.2f}s"
            for sp in stage_points
        )
        print(f"  {alg_status}")

    summary_rows: List[Dict[str, object]] = []
    for sp in stage_points:
        acc_mu, acc_sd = mean_std(row[f"alg3_test_acc_stage_{sp}"] for row in per_seed_rows)
        t_mu, t_sd = mean_std(row[f"alg3_train_time_s_stage_{sp}"] for row in per_seed_rows)
        k_mu, k_sd = mean_std(row[f"alg3_selected_size_stage_{sp}"] for row in per_seed_rows)
        summary_rows.append({
            "method": f"Algorithm 3 (stage {sp})",
            "B": args.B,
            "S-junta": args.S,
            "K": args.K,
            "train_size": args.train_size,
            "test_size": len(x_test),
            "test_accuracy_mean": acc_mu,
            "test_accuracy_std": acc_sd,
            "test_accuracy_mean_pm_std": fmt_pm(acc_mu, acc_sd),
            "train_runtime_s_mean": t_mu,
            "train_runtime_s_std": t_sd,
            "train_runtime_s_mean_pm_std": fmt_pm(t_mu, t_sd, digits=2),
            "selected_size_mean": k_mu,
            "selected_size_std": k_sd,
            "selected_size_mean_pm_std": fmt_pm(k_mu, k_sd, digits=2),
        })

    summary_path = outdir / "algorithm3_stage_summary.csv"
    with summary_path.open("w", newline="") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    # Ablation plot: effect of number of residual stages on test accuracy.
    # The shaded band is a normal-approximation confidence interval over seeds:
    # mean +/- z * std / sqrt(number of seeds), with z=1.96 by default.
    plot_path: Optional[Path] = None
    if args.make_stage_plot and plot_stage_points:
        xs: List[int] = []
        means: List[float] = []
        lowers: List[float] = []
        uppers: List[float] = []
        for sp in plot_stage_points:
            vals = [float(row[f"alg3_test_acc_stage_{sp}"]) for row in per_seed_rows]
            vals = [v for v in vals if v == v]
            if not vals:
                continue
            mu = statistics.mean(vals)
            sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
            half_width = args.ci_z * sd / math.sqrt(len(vals))
            xs.append(sp)
            means.append(mu)
            lowers.append(max(0.0, mu - half_width))
            uppers.append(min(1.0, mu + half_width))

        if xs:
            fig, ax = plt.subplots(figsize=(6.5, 4.2))
            ax.plot(xs, means, marker="o", linewidth=2)
            ax.fill_between(xs, lowers, uppers, alpha=0.2)
            ax.set_xlabel("Number of residual stages")
            ax.set_ylabel("Test accuracy")
            ax.set_title("Algorithm 3: test accuracy over residual stages")
            ax.set_ylim(0.6, 1.02)
            ax.set_xticks(xs)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            plot_path = outdir / "algorithm3_accuracy_over_stages.png"
            fig.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close(fig)

    print("\n=== Algorithm 3 summary: mean +- sample std over seeds ===")
    header = f"{'Stage':>8s} | {'Test acc':17s} | {'Train runtime [s]':20s} | {'Selected |J|':14s}"
    print(header)
    print("-" * len(header))
    for r in summary_rows:
        stage = str(r["method"]).replace("Algorithm 3 (stage ", "").replace(")", "")
        print(
            f"{stage:>8s} | "
            f"{r['test_accuracy_mean_pm_std']:17s} | "
            f"{r['train_runtime_s_mean_pm_std']:20s} | "
            f"{r['selected_size_mean_pm_std']:14s}"
        )

    print(f"\nSaved Algorithm 3 stage summary to: {summary_path}")
    if plot_path is not None:
        print(f"Saved ablation plot to: {plot_path}")
    return summary_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--B", type=int, default=12, help="ambient input dimension")
    p.add_argument("--S", type=int, default=8, help="number of relevant variables in random S-junta")
    p.add_argument("--train-size", type=int, default=1000)
    p.add_argument("--test-size", type=int, default=2**12)
    p.add_argument("--full-cube-test", action=argparse.BooleanOptionalAction, default=False,
                   help="use the full Boolean cube as test set; overrides --test-size")
    p.add_argument("--K", type=int, default=6, help="top-K selected variables per residual stage")
    p.add_argument("--tau", type=float, default=0.0, help="influence threshold")
    p.add_argument("--stages", type=int, default=20, help="number of residual stages")
    p.add_argument("--stage-points", type=int, nargs="+", default=[1, 5, 20],
                   help="stages at which to report Algorithm 3 accuracy/runtime")
    p.add_argument("--num-seeds", type=int, default=20)
    p.add_argument("--base-seed", type=int, default=0)
    p.add_argument("--outdir", type=str, default="results_alg3_only")
    p.add_argument("--make-stage-plot", action=argparse.BooleanOptionalAction, default=True,
                   help="save ablation plot: test accuracy vs number of residual stages")
    p.add_argument("--plot-every-stage", action=argparse.BooleanOptionalAction, default=True,
                   help="plot stages 1,2,...,--stages; use --no-plot-every-stage to plot only --plot-stage-points")
    p.add_argument("--plot-stage-points", type=int, nargs="+", default=None,
                   help="stages to use in the plot when --no-plot-every-stage is set; defaults to --stage-points")
    p.add_argument("--ci-z", type=float, default=1.96,
                   help="z value for shaded confidence interval; 1.96 gives approximate 95%% CI")
    return p.parse_args()


if __name__ == "__main__":
    run_experiment(parse_args())
