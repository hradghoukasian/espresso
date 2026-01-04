# row_partition_softmin_vote.py
import random
import math
import statistics
import time
from typing import List, Dict, Tuple, Optional

from espresso_minimization import minimize_truth_table_espresso, build_tt_from_on_off_sets


# ----------------------------
# Target function generators
# ----------------------------

def random_truth_table(B: int, seed: int = 0) -> List[int]:
    rng = random.Random(seed)
    return [rng.getrandbits(1) for _ in range(1 << B)]


def random_s_junta_truth_table(B: int, S: int, seed: int = 0) -> Tuple[List[int], List[int]]:
    rng = random.Random(seed)
    J = sorted(rng.sample(range(B), S))
    h_tt = [rng.getrandbits(1) for _ in range(1 << S)]

    f_tt = [0] * (1 << B)
    for x in range(1 << B):
        u = 0
        for k, bitpos in enumerate(J):
            if (x >> bitpos) & 1:
                u |= (1 << k)
        f_tt[x] = h_tt[u]
    return f_tt, J


# ----------------------------
# Partition of rows (disjoint)
# ----------------------------

def random_partition_indices(size: int, N: int, seed: int = 0) -> Tuple[List[List[int]], List[int]]:
    """
    Randomly partitions {0,...,size-1} into N disjoint blocks (balanced round-robin).
    Returns:
      parts[n] = list of indices assigned to block n
      owner[x] = n such that x belongs to block n
    """
    if N <= 0:
        raise ValueError("N must be positive")
    if N > size:
        raise ValueError("N cannot exceed size")

    rng = random.Random(seed)
    perm = list(range(size))
    rng.shuffle(perm)

    parts = [[] for _ in range(N)]
    owner = [-1] * size
    for i, x in enumerate(perm):
        n = i % N
        parts[n].append(x)
        owner[x] = n

    return parts, owner


# ----------------------------
# Espresso helpers
# ----------------------------

def build_partial_tt_from_rows(B: int, f_tt: List[int], rows: List[int]) -> str:
    """
    Build a length-2^B truth table string over {0,1,-}:
      - specified rows get their true label (0/1)
      - all other rows are '-' (don't care)
    """
    onset = [idx for idx in rows if f_tt[idx] == 1]
    offset = [idx for idx in rows if f_tt[idx] == 0]
    return build_tt_from_on_off_sets(B, onset, offset, default="-")


def eval_expr_on_int_with_vars(expr, vars_list, x: int) -> int:
    """
    Evaluate PyEDA expression on input x (integer).
    vars_list should be stats["variables"] = [x0, ..., x_{B-1}]
    """
    assignment = {vars_list[i]: (x >> i) & 1 for i in range(len(vars_list))}
    val = expr.restrict(assignment)
    return 1 if val.is_one() else 0


# ----------------------------
# Metric rho + softmin
# ----------------------------

def hamming_distance_int(x: int, z: int) -> int:
    return (x ^ z).bit_count()


def softmin(distances: List[float], beta: float) -> List[float]:
    """
    Softmin with temperature beta:
      w_n = exp(-beta d_n) / sum_j exp(-beta d_j)
    """
    m = min(distances)
    exps = [math.exp(-beta * (d - m)) for d in distances]  # stable
    Z = sum(exps)
    return [e / Z for e in exps]


# ----------------------------
# Training: partitioned rows, N circuits
# ----------------------------

def train_partitioned_row_ensemble(
    B: int,
    N: int,
    seed: int = 0,
    target: str = "random",      # "random" or "junta"
    junta_S: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[List[int], List[Dict], List[List[int]], Optional[List[int]], List[float]]:
    """
    Train N circuits C_n on disjoint row blocks (a partition).
    Each circuit sees only its block rows; others are don't-care '-'.

    Returns:
      f_tt, ensemble, parts, J, train_times
      where train_times[n] is the training time (seconds) for circuit n.
    """
    size = 1 << B

    # target truth table
    J = None
    if target == "random":
        f_tt = random_truth_table(B, seed=seed)
        if verbose:
            print(f"[Target] random truth table on B={B}")
    elif target == "junta":
        if junta_S is None:
            raise ValueError("Provide junta_S when target='junta'")
        f_tt, J = random_s_junta_truth_table(B, junta_S, seed=seed)
        if verbose:
            print(f"[Target] random {junta_S}-junta on B={B}, J={J}")
    else:
        raise ValueError("target must be 'random' or 'junta'")

    # disjoint partition of rows (seeded)
    parts, _owner = random_partition_indices(size=size, N=N, seed=seed + 999)
    if verbose:
        sizes = [len(p) for p in parts]
        print(f"[Partition] N={N}, block sizes: min={min(sizes)}, max={max(sizes)}, mean={sum(sizes)/N:.2f}")

    ensemble: List[Dict] = []
    train_times: List[float] = []

    for n in range(N):
        rows_n = parts[n]
        if verbose:
            print(f"[Training] Circuit {n+1}/{N}: |rows|={len(rows_n)}")

        t0 = time.perf_counter()

        tt_partial = build_partial_tt_from_rows(B, f_tt, rows_n)
        expr, stats = minimize_truth_table_espresso(tt_partial, verbose=False)

        elapsed = time.perf_counter() - t0
        train_times.append(elapsed)

        ensemble.append({
            "rows": rows_n,
            "expr": expr,
            "stats": stats,
        })

    if verbose:
        print("[Training] All circuits trained.")

    return f_tt, ensemble, parts, J, train_times


# ----------------------------
# Inference:
# ----------------------------

def attention_weights_for_x(
    x: int,
    ensemble: List[Dict],
    beta: float,
) -> List[float]:
    """
    d_n(x) = min_{z in rows_n} rho(x, z)
    w(x) = softmin(d_1(x),...,d_N(x))
    """
    distances = []
    for item in ensemble:
        rows = item["rows"]
        d = min(hamming_distance_int(x, z) for z in rows)
        distances.append(d)
    return softmin(distances, beta=beta)


def predict_softmin_threshold_majority(
    x: int,
    ensemble: List[Dict],
    beta: float,
    lam: float,
    tie_break: int = 1,
    fallback_all_if_empty: bool = True,
) -> int:
    """

      w(x) = Softmin( [min_m rho(x, R_{n:m})]_{n=1..N} )
      C(x) = Maj( { C_n(x) : w_n(x) >= lam } )

    If no circuit passes the threshold and fallback_all_if_empty=True,
    we vote over all circuits (practical safeguard).
    """
    w = attention_weights_for_x(x, ensemble, beta=beta)
    active = [n for n in range(len(ensemble)) if w[n] >= lam]

    if not active and fallback_all_if_empty:
        active = list(range(len(ensemble)))

    votes = 0
    for n in active:
        item = ensemble[n]
        y = eval_expr_on_int_with_vars(item["expr"], item["stats"]["variables"], x)
        votes += y

    if 2 * votes > len(active):
        return 1
    if 2 * votes < len(active):
        return 0
    return 1 if tie_break == 1 else 0


def evaluate_full_tt(
    f_tt: List[int],
    ensemble: List[Dict],
    B: int,
    beta: float,
    lam: float,
    verbose: bool = False,
    print_every: int = 200000,
) -> float:
    size = 1 << B
    correct = 0
    for x in range(size):
        if verbose and x > 0 and (x % print_every == 0):
            print(f"[Eval] {x}/{size} ...")

        y_hat = predict_softmin_threshold_majority(x, ensemble, beta=beta, lam=lam)
        if y_hat == f_tt[x]:
            correct += 1
    return correct / size


# ----------------------------
# Main (10 seeds)
# ----------------------------

def main():
    # Hyperparameters
    B = 12
    N = 1000

    beta = 1.0
    lam = 0.02

    target = "random"   # or "junta"
    junta_S = 4         # only used if target="junta"

    num_seeds = 10
    accuracies: List[float] = []
    avg_train_times_per_seed: List[float] = []
    total_times_per_seed: List[float] = []   # NEW

    for seed in range(num_seeds):
        print("\n" + "=" * 60)
        print(f"[Run] Seed {seed}")
        print("=" * 60)

        # ----------------------------
        # Start total timer for seed
        # ----------------------------
        seed_start = time.perf_counter()

        f_tt, ensemble, parts, J, train_times = train_partitioned_row_ensemble(
            B=B,
            N=N,
            seed=seed,
            target=target,
            junta_S=junta_S,
            verbose=False,
        )

        acc = evaluate_full_tt(
            f_tt=f_tt,
            ensemble=ensemble,
            B=B,
            beta=beta,
            lam=lam,
            verbose=False,
        )

        # ----------------------------
        # End total timer for seed
        # ----------------------------
        seed_elapsed = time.perf_counter() - seed_start
        total_times_per_seed.append(seed_elapsed)

        # Espresso-only timing (already present)
        avg_time = sum(train_times) / len(train_times)
        avg_train_times_per_seed.append(avg_time)

        print(f"[Timing] Avg Espresso training time per subcircuit: {avg_time:.6f} s")
        print(f"[Timing] Total runtime for seed {seed}: {seed_elapsed:.6f} s")
        print(f"[Result] Seed {seed} accuracy = {acc:.6f}")

        accuracies.append(acc)

    # ----------------------------
    # Aggregate statistics
    # ----------------------------
    mean_acc = statistics.mean(accuracies)
    std_acc = statistics.stdev(accuracies) if num_seeds > 1 else 0.0

    mean_espresso_time = statistics.mean(avg_train_times_per_seed)
    std_espresso_time = statistics.stdev(avg_train_times_per_seed) if num_seeds > 1 else 0.0

    mean_total_time = statistics.mean(total_times_per_seed)
    std_total_time = statistics.stdev(total_times_per_seed) if num_seeds > 1 else 0.0

    print("\n" + "=" * 60)
    print("Final results over seeds")
    print("=" * 60)
    print(
        f"B={B}, N={N}, beta={beta}, lam={lam}, target={target}"
        + (f", S={junta_S}" if target == "junta" else "")
    )
    print(f"Accuracy (mean ± std): {mean_acc:.6f} ± {std_acc:.6f}")
    print(
        f"Avg Espresso time per subcircuit (mean ± std): "
        f"{mean_espresso_time:.6f} ± {std_espresso_time:.6f} s"
    )
    print(
        f"Total runtime per seed (mean ± std): "
        f"{mean_total_time:.6f} ± {std_total_time:.6f} s"
    )

if __name__ == "__main__":
    main()
