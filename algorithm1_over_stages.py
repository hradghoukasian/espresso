"""
algorithm1_accuracy_over_stages.py

Experiment: run Algorithm 1 (sample-based multi-stage influence residual learning
on a partial truth table) and report test accuracy over residual stages.


Ablation plot is generated: test accuracy vs number of residual stages.
Training runtime is cumulative through each reported stage and excludes test
accuracy evaluation time.

"""
from __future__ import annotations

import csv
import math
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from pyeda.inter import exprvars, truthtable
from pyeda.boolalg.minimization import espresso_tts


# ============================================================
# CONFIGURATION: edit only this block
# ============================================================

CONFIG_NAME = "Config. 11"
OUTDIR = "results_config11"
S = 10
B = 21
TRAIN_SIZE = 2**21
TEST_SIZE = 2**17
K = 8


TAU = 0.0
STAGES = 20
NUM_SEEDS = 20
BASE_SEED = 0



# 95% Student-t CI for n=20:
# t_{0.025,19} = 2.093
T_CRIT = 2.093


# ============================================================
# Utilities
# ============================================================

def proj_index(x: int, cols: Sequence[int]) -> int:
    u = 0
    for j, bit in enumerate(cols):
        if (x >> bit) & 1:
            u |= 1 << j
    return u


@dataclass
class SJunta:
    bits: List[int]
    tt: List[int]

    def __call__(self, x: int) -> int:
        return self.tt[proj_index(x, self.bits)]


def make_random_junta(B: int, S: int, rng: random.Random) -> SJunta:
    bits = sorted(rng.sample(range(B), S))
    tt = [rng.getrandbits(1) for _ in range(1 << S)]
    return SJunta(bits, tt)


def sample_dataset(f, B: int, n: int, rng: random.Random):
    xs = rng.sample(range(1 << B), n)
    ys = [f(x) for x in xs]
    return xs, ys


# ============================================================
# Espresso
# ============================================================

@dataclass
class Stage:
    bits: List[int]
    tt: List[int]

    def predict(self, x: int) -> int:
        return self.tt[proj_index(x, self.bits)]


def predict(stages: Sequence[Stage], x: int) -> int:
    y = 0
    for stage in stages:
        y ^= stage.predict(x)
    return y


def espresso_learn(tt: Sequence[str], n_vars: int) -> List[int]:
    size = 1 << n_vars
    specified = {v for v in tt if v in {"0", "1"}}

    if not specified or specified == {"0"}:
        return [0] * size

    if specified == {"1"}:
        return [1] * size

    X = exprvars("x", n_vars)
    f_tt = truthtable(X, "".join(tt))
    expr, = espresso_tts(f_tt)

    output = []

    for u in range(size):
        assignment = {
            X[j]: ((u >> j) & 1)
            for j in range(n_vars)
        }

        val = expr.restrict(assignment)
        output.append(1 if val.is_one() else 0)

    return output


# ============================================================
# Algorithm 1
# ============================================================

def influences(
    B: int,
    x_train: Sequence[int],
    residual_map: Dict[int, int],
) -> List[float]:

    xset = set(x_train)
    result = [0.0] * B

    for i in range(B):
        mask = 1 << i
        mismatches = 0
        count = 0

        for x in x_train:

            # Count each undirected edge once
            if ((x >> i) & 1) != 0:
                continue

            xn = x ^ mask

            if xn in xset:
                count += 1
                mismatches += (
                    residual_map[x] != residual_map[xn]
                )

        result[i] = mismatches / count if count else 0.0

    return result


def select_top_k(
    infl: Sequence[float],
    rng: random.Random,
) -> List[int]:

    candidates = [
        i for i, value in enumerate(infl)
        if value > TAU
    ]

    rng.shuffle(candidates)

    candidates.sort(
        key=lambda i: infl[i],
        reverse=True,
    )

    return candidates[:K]


def build_projected_tt(
    x_train: Sequence[int],
    residuals: Sequence[int],
    J: Sequence[int],
) -> List[str]:

    size = 1 << len(J)

    zeros = [0] * size
    ones = [0] * size

    for x, r in zip(x_train, residuals):
        u = proj_index(x, J)

        if r:
            ones[u] += 1
        else:
            zeros[u] += 1

    output = []

    for u in range(size):

        if ones[u] > zeros[u]:
            output.append("1")

        elif zeros[u] > ones[u]:
            output.append("0")

        else:
            # tie or unseen -> don't care
            output.append("-")

    return output


def accuracy(
    stages: Sequence[Stage],
    xs: Sequence[int],
    ys: Sequence[int],
) -> float:

    correct = sum(
        predict(stages, x) == y
        for x, y in zip(xs, ys)
    )

    return correct / len(xs)


def run_one_seed(seed: int):

    rng = random.Random(seed)

    f = make_random_junta(B, S, rng)

    x_train, y_train = sample_dataset(
        f, B, TRAIN_SIZE, rng
    )

    x_test, y_test = sample_dataset(
        f, B, TEST_SIZE, rng
    )

    stages = []

    acc_by_stage = {}
    time_by_stage = {}

    cumulative_time = 0.0

    for t in range(1, STAGES + 1):

        start = time.perf_counter()

        residuals = [
            y ^ predict(stages, x)
            for x, y in zip(x_train, y_train)
        ]

        # Stop if residual is zero
        if all(r == 0 for r in residuals):
            cumulative_time += time.perf_counter() - start

            final_acc = accuracy(
                stages,
                x_test,
                y_test,
            )

            for s in range(t, STAGES + 1):
                acc_by_stage[s] = final_acc
                time_by_stage[s] = cumulative_time

            break

        residual_map = dict(
            zip(x_train, residuals)
        )

        infl = influences(
            B,
            x_train,
            residual_map,
        )

        J = select_top_k(
            infl,
            rng,
        )

        # Empty active set
        if not J:
            cumulative_time += time.perf_counter() - start

            final_acc = accuracy(
                stages,
                x_test,
                y_test,
            )

            for s in range(t, STAGES + 1):
                acc_by_stage[s] = final_acc
                time_by_stage[s] = cumulative_time

            break

        tt = build_projected_tt(
            x_train,
            residuals,
            J,
        )

        stage_tt = espresso_learn(
            tt,
            len(J),
        )

        stages.append(
            Stage(
                bits=J,
                tt=stage_tt,
            )
        )

        cumulative_time += time.perf_counter() - start

        acc_by_stage[t] = accuracy(
            stages,
            x_test,
            y_test,
        )

        time_by_stage[t] = cumulative_time

    return acc_by_stage, time_by_stage


# ============================================================
# Statistics
# ============================================================

def mean_std_ci(values):

    mu = statistics.mean(values)
    sd = statistics.stdev(values)

    half_width = (
        T_CRIT
        * sd
        / math.sqrt(NUM_SEEDS)
    )

    return (
        mu,
        sd,
        mu - half_width,
        mu + half_width,
    )


# ============================================================
# Main
# ============================================================

def main():

    outdir = Path(OUTDIR)
    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    for i in range(NUM_SEEDS):

        seed = BASE_SEED + i

        print(
            f"{CONFIG_NAME}, seed {seed}"
        )

        acc_by_stage, time_by_stage = run_one_seed(
            seed
        )

        for stage in range(1, STAGES + 1):

            rows.append({
                "config": CONFIG_NAME,
                "seed": seed,
                "stage": stage,
                "accuracy": acc_by_stage[stage],
                "runtime": time_by_stage[stage],
            })

    # --------------------------------------------------------
    # Save per-seed results
    # --------------------------------------------------------

    per_seed_file = (
        outdir
        / "stage_per_seed.csv"
    )

    with per_seed_file.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(rows)

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    summary = []

    for stage in range(1, STAGES + 1):

        stage_rows = [
            r for r in rows
            if r["stage"] == stage
        ]

        values = [
            r["accuracy"]
            for r in stage_rows
        ]

        runtimes = [
            r["runtime"]
            for r in stage_rows
        ]

        mu, sd, lo, hi = mean_std_ci(
            values
        )

        summary.append({
            "config": CONFIG_NAME,
            "stage": stage,
            "accuracy_mean": mu,
            "accuracy_std": sd,
            "ci95_lower": lo,
            "ci95_upper": hi,
            "runtime_mean": statistics.mean(runtimes),
            "runtime_std": statistics.stdev(runtimes),
        })

    summary_file = (
        outdir
        / "stage_summary.csv"
    )

    with summary_file.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=summary[0].keys(),
        )

        writer.writeheader()
        writer.writerows(summary)

    # --------------------------------------------------------
    # Print stages 1, 5, 20
    # --------------------------------------------------------

    print()

    for stage in [1, 5, 20]:

        row = summary[stage - 1]

        print(
            f"Stage {stage}: "
            f"{row['accuracy_mean']:.3f} "
            f"+- {row['accuracy_std']:.3f}, "
            f"95% CI = "
            f"[{row['ci95_lower']:.3f}, "
            f"{row['ci95_upper']:.3f}]"
        )

    print()
    print("Saved:")
    print(per_seed_file)
    print(summary_file)


if __name__ == "__main__":
    main()