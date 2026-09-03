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


def require_pyeda() -> None:
    if PYEDA_IMPORT_ERROR is not None:
        raise RuntimeError(
            "PyEDA is required for the actual Espresso calls. Install it with:\n"
            "    pip install pyeda\n"
            f"Original import error: {PYEDA_IMPORT_ERROR!r}"
        )


def proj_index(x: int, cols: Sequence[int]) -> int:
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
    correct = sum(int(int(predict_fn(x)) == int(y)) for x, y in zip(xs, ys))
    return correct / len(xs)


@dataclass
class EspressoModel:
    n_vars: int
    expr: object = None
    constant_value: Optional[int] = None
    tt_full: Optional[List[int]] = None
    variables: object = None


def espresso_learn_tt(tt: Sequence[str], n_vars: int) -> EspressoModel:
    require_pyeda()
    if len(tt) != (1 << n_vars):
        raise ValueError(f"len(tt)={len(tt)} but expected 2^{n_vars}.")

    chars = list(tt)
    values = set(chars)
    if not values.issubset({"0", "1", "-", "x"}):
        raise ValueError(f"Truth table has invalid characters: {values}.")

    size = 1 << n_vars
    specified = {c for c in chars if c in {"0", "1"}}

    if n_vars == 0:
        value = 1 if chars[0] == "1" else 0
        return EspressoModel(n_vars=0, constant_value=value, tt_full=[value])
    if not specified or specified == {"0"}:
        return EspressoModel(n_vars=n_vars, constant_value=0, tt_full=[0] * size)
    if specified == {"1"}:
        return EspressoModel(n_vars=n_vars, constant_value=1, tt_full=[1] * size)

    X = exprvars("x", n_vars)
    f_tt = truthtable(X, "".join(chars))
    expr, = espresso_tts(f_tt)

    full_predictions: List[int] = []
    for u in range(size):
        assignment = {X[j]: ((u >> j) & 1) for j in range(n_vars)}
        val = expr.restrict(assignment)
        full_predictions.append(1 if val.is_one() else 0)

    return EspressoModel(
        n_vars=n_vars,
        expr=expr,
        tt_full=full_predictions,
        variables=X,
    )


@dataclass
class Stage:
    bits: List[int]
    tt_full: List[int]

    def predict_one(self, x: int) -> int:
        return int(self.tt_full[proj_index(x, self.bits)])


def predict_stages(stages: Sequence[Stage], x: int) -> int:
    y = 0
    for st in stages:
        y ^= st.predict_one(x)
    return y


def select_random_k(B: int, K: int, rng: random.Random) -> List[int]:
    k = min(K, B)
    return sorted(rng.sample(range(B), k))


def projected_residual_partial_tt(
    x_train: Sequence[int],
    r_train: Sequence[int],
    J: Sequence[int],
) -> List[str]:
    k = len(J)
    if k == 0:
        ones = sum(r_train)
        zeros = len(r_train) - ones
        if ones > zeros:
            return ["1"]
        if zeros > ones:
            return ["0"]
        return ["-"]

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
        if ones[u] > zeros[u]:
            out.append("1")
        elif zeros[u] > ones[u]:
            out.append("0")
        else:
            out.append("-")
    return out


def train_random_k_residual(
    B: int,
    K: int,
    stages_m: int,
    x_train: Sequence[int],
    y_train: Sequence[int],
    seed: int,
) -> Tuple[List[Stage], float, List[int], int]:
    rng = random.Random(seed)
    stages: List[Stage] = []
    selected_sizes: List[int] = []
    cumulative_train_time = 0.0
    T_stop = 0

    for t in range(1, stages_m + 1):
        stage_start = time.perf_counter()

        r_train = [
            int(y) ^ predict_stages(stages, x)
            for x, y in zip(x_train, y_train)
        ]

        if all(r == 0 for r in r_train):
            cumulative_train_time += time.perf_counter() - stage_start
            break

        J = select_random_k(B=B, K=K, rng=rng)
        tt_proj = projected_residual_partial_tt(x_train, r_train, J)
        model = espresso_learn_tt(tt_proj, n_vars=len(J))

        if model.tt_full is None:
            raise RuntimeError("Expected cached projected truth table.")

        stages.append(Stage(bits=list(J), tt_full=list(model.tt_full)))
        selected_sizes.append(len(J))
        T_stop = t
        cumulative_train_time += time.perf_counter() - stage_start

    return stages, cumulative_train_time, selected_sizes, T_stop


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

        # Keep the same random-K RNG convention as the prior comparison script.
        random_k_seed = 20_000_000 + seed

        stages_rand, time_rand, sizes_rand, stop_rand = train_random_k_residual(
            B=args.B,
            K=args.K,
            stages_m=args.stages,
            x_train=x_train,
            y_train=y_train,
            seed=random_k_seed,
        )

        acc_rand = accuracy_from_predictor(
            lambda z: predict_stages(stages_rand, z),
            x_test,
            y_test,
        )

        print(
            f"  Random K: acc={acc_rand:.4f}, "
            f"train_time={time_rand:.2f}s, T_stop={stop_rand}"
        )

        per_seed_rows.append({
            "seed": seed,
            "B": args.B,
            "S": args.S,
            "K": args.K,
            "train_size": args.train_size,
            "test_size": test_size_used,
            "stages": args.stages,
            "junta_bits": " ".join(map(str, f.junta_bits)),
            "random_final_test_acc": acc_rand,
            "random_train_time_s": time_rand,
            "random_mean_selected_size": statistics.mean(sizes_rand) if sizes_rand else float("nan"),
            "random_stopping_stage": stop_rand,
        })

    acc_mu, acc_sd = mean_std(row["random_final_test_acc"] for row in per_seed_rows)
    t_mu, t_sd = mean_std(row["random_train_time_s"] for row in per_seed_rows)
    k_mu, k_sd = mean_std(row["random_mean_selected_size"] for row in per_seed_rows)
    stop_mu, stop_sd = mean_std(row["random_stopping_stage"] for row in per_seed_rows)

    summary_row = {
        "method": "Algorithm 1 (random K ablation)",
        "B": args.B,
        "S-junta": args.S,
        "K": args.K,
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
        "stopping_stage_mean": stop_mu,
        "stopping_stage_std": stop_sd,
    }

    summary_path = outdir / "alg3_randomK_only_summary.csv"
    with summary_path.open("w", newline="") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=list(summary_row.keys()))
        writer.writeheader()
        writer.writerow(summary_row)

    print("\n=== Summary: mean +- sample std over seeds ===")
    print(
        f"Random K | Test acc: {summary_row['final_test_accuracy_mean_pm_std']} | "
        f"Train runtime [s]: {summary_row['train_runtime_s_mean_pm_std']} | "
        f"Mean T_stop: {stop_mu:.2f} +- {stop_sd:.2f}"
    )

    print(f"\nSaved summary table to: {summary_path}")
    return summary_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--B", type=int, default=20, help="ambient input dimension")
    p.add_argument("--S", type=int, default=8, help="number of relevant variables in random S-junta")
    p.add_argument("--train-size", type=int, default=2**20)
    p.add_argument("--test-size", type=int, default=2**17, help="sampled test size, ignored if --full-cube-test is used")
    p.add_argument("--full-cube-test", action="store_true", help="evaluate on the whole Boolean cube")
    p.add_argument("--K", type=int, default=6, help="number of randomly selected variables per residual stage")
    p.add_argument("--stages", type=int, default=20, help="maximum number of residual stages")
    p.add_argument("--num-seeds", type=int, default=20)
    p.add_argument("--base-seed", type=int, default=0)
    p.add_argument("--outdir", type=str, default="results_alg3_randomK_only")
    return p.parse_args()


if __name__ == "__main__":
    run_experiment(parse_args())