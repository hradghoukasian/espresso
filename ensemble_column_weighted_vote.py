# ensemble_column_weighted_vote.py
import random
import math
import statistics
from typing import List, Dict, Optional, Tuple

# Only needed if we set run_espresso=True
from espresso_minimization import minimize_truth_table_espresso


# ----------------------------
# Target function generators
# ----------------------------

def random_truth_table(B: int, seed: int = 0) -> List[int]:
    rng = random.Random(seed)
    size = 1 << B
    return [rng.getrandbits(1) for _ in range(size)]


def random_s_junta_truth_table(
    B: int,
    S: int,
    seed: int = 0,
    relevant_set: Optional[List[int]] = None,
) -> Tuple[List[int], List[int]]:
    if not (0 <= S <= B):
        raise ValueError("Require 0 <= S <= B")

    rng = random.Random(seed)

    if relevant_set is None:
        J = sorted(rng.sample(range(B), S))
    else:
        if len(relevant_set) != S:
            raise ValueError("relevant_set must have length S")
        if len(set(relevant_set)) != S or any(j < 0 or j >= B for j in relevant_set):
            raise ValueError("relevant_set must be S distinct indices in [0,B)")
        J = sorted(relevant_set)

    h_tt = [rng.getrandbits(1) for _ in range(1 << S)]

    size_B = 1 << B
    f_tt = [0] * size_B
    for x in range(size_B):
        u = 0
        for k, bitpos in enumerate(J):
            if (x >> bitpos) & 1:
                u |= (1 << k)
        f_tt[x] = h_tt[u]

    return f_tt, J


# ----------------------------
# Projection and learning
# ----------------------------

def proj_index(x: int, cols: List[int]) -> int:
    u = 0
    for k, bitpos in enumerate(cols):
        if (x >> bitpos) & 1:
            u |= (1 << k)
    return u


def build_projected_tt_majority(
    f_tt: List[int],
    B: int,
    cols: List[int],
    tie_break: int = 1,
) -> str:
    b = len(cols)
    size_B = 1 << B
    size_b = 1 << b

    ones = [0] * size_b
    total = [0] * size_b

    for x in range(size_B):
        u = proj_index(x, cols)
        ones[u] += f_tt[x]
        total[u] += 1

    g = []
    for u in range(size_b):
        if 2 * ones[u] > total[u]:
            g.append("1")
        elif 2 * ones[u] < total[u]:
            g.append("0")
        else:
            g.append("1" if tie_break == 1 else "0")

    return "".join(g)


def coverage_probability(B: int, b: int, S: int) -> float:
    if b < S:
        return 0.0
    return math.comb(B - S, b - S) / math.comb(B, b)


def train_ensemble_column_only(
    B: int,
    b: int,
    N: int,
    seed: int = 0,
    tie_break_proj: int = 1,
    run_espresso: bool = False,
    verbose: bool = True,
    target: str = "random",          # "random" or "junta"
    junta_S: Optional[int] = None,   # used iff target="junta"
    relevant_set: Optional[List[int]] = None,
) -> Tuple[List[int], List[Dict], Optional[List[int]]]:
    if not (0 < b <= B):
        raise ValueError("Require 0 < b <= B")

    J: Optional[List[int]] = None
    if target == "random":
        f_tt = random_truth_table(B, seed=seed)
        if verbose:
            print(f"[Target] random truth table on B={B} bits")
    elif target == "junta":
        if junta_S is None:
            raise ValueError("When target='junta', you must provide junta_S")
        f_tt, J = random_s_junta_truth_table(B, junta_S, seed=seed, relevant_set=relevant_set)
        if verbose:
            print(f"[Target] random {junta_S}-junta on B={B} bits, relevant bits J={J}")
            p = coverage_probability(B, b, junta_S)
            print(f"[Target] P(J subset of random b-subset) = {p:.6f} (expected hits among N: {N*p:.2f})")
    else:
        raise ValueError("target must be 'random' or 'junta'")

    rng = random.Random(seed + 12345)

    ensemble: List[Dict] = []
    for n in range(N):
        cols = sorted(rng.sample(range(B), b))

        overlap = None
        if J is not None:
            overlap = len(set(cols).intersection(J))

        if verbose:
            if overlap is None:
                print(f"[Training] Subcircuit {n+1}/{N} (B={B}, b={b})")
            else:
                print(f"[Training] Subcircuit {n+1}/{N} (B={B}, b={b}) | overlap={overlap}/{len(J)}")

        g_tt = build_projected_tt_majority(f_tt, B, cols, tie_break=tie_break_proj)

        stats = None
        if run_espresso:
            _, stats = minimize_truth_table_espresso(g_tt, verbose=False)

        ensemble.append({
            "cols": cols,
            "g_tt": g_tt,
            "espresso_stats": stats,
            "overlap": overlap,
        })

    if verbose:
        print("[Training] All subcircuits trained.")

    return f_tt, ensemble, J


# ----------------------------
# Evaluation + weights
# ----------------------------

def subcircuit_accuracy_full_tt(f_tt: List[int], B: int, item: Dict) -> float:
    """
    Accuracy of one subcircuit over the full truth table.
    item must have keys: "cols", "g_tt"
    """
    size_B = 1 << B
    cols = item["cols"]
    g_tt = item["g_tt"]

    correct = 0
    for x in range(size_B):
        u = proj_index(x, cols)
        y_hat = 1 if g_tt[u] == "1" else 0
        if y_hat == f_tt[x]:
            correct += 1
    return correct / size_B


def assign_centered_weights(
    f_tt: List[int],
    ensemble: List[Dict],
    B: int,
    center: float = 0.5,
) -> List[float]:
    """
    Compute weights w_n = max(Acc_n - center, 0).
    Stores each subcircuit's accuracy in item["acc"] for convenience.
    """
    weights: List[float] = []
    for item in ensemble:
        acc = subcircuit_accuracy_full_tt(f_tt, B, item)
        item["acc"] = acc
        w = max(acc - center, 0.0)
        item["weight"] = w
        weights.append(w)
    return weights


def ensemble_predict_weighted(
    x: int,
    ensemble: List[Dict],
    tie_break_vote: int = 1,
) -> int:
    """
    Weighted vote:
        score = sum_n w_n * y_n(x)
        predict 1 if score >= 0.5 * sum_n w_n
    If all weights are 0, fall back to unweighted majority vote.
    """
    total_w = 0.0
    score = 0.0

    for item in ensemble:
        w = item.get("weight", 0.0)
        total_w += w
        if w == 0.0:
            continue
        u = proj_index(x, item["cols"])
        y = 1.0 if item["g_tt"][u] == "1" else 0.0
        score += w * y

    # If everything got weight 0 (all <= chance), fall back to plain majority.
    if total_w == 0.0:
        votes = 0
        N = len(ensemble)
        for item in ensemble:
            u = proj_index(x, item["cols"])
            votes += 1 if item["g_tt"][u] == "1" else 0
        if 2 * votes > N:
            return 1
        if 2 * votes < N:
            return 0
        return 1 if tie_break_vote == 1 else 0

    # Weighted threshold at half the total weight
    if score > 0.5 * total_w:
        return 1
    if score < 0.5 * total_w:
        return 0
    return 1 if tie_break_vote == 1 else 0


def evaluate_ensemble_full_tt_weighted(
    f_tt: List[int],
    ensemble: List[Dict],
    B: int,
    tie_break_vote: int = 1,
    verbose: bool = False,
    print_every: int = 200000,
) -> float:
    size_B = 1 << B
    correct = 0
    for x in range(size_B):
        if verbose and (x % print_every == 0) and x > 0:
            print(f"[Eval] {x}/{size_B} ...")

        y_hat = ensemble_predict_weighted(x, ensemble, tie_break_vote=tie_break_vote)
        if y_hat == f_tt[x]:
            correct += 1
    return correct / size_B


# ----------------------------
# Main: 10-seed experiment
# ----------------------------

def main():
    # ---- Experiment params ----
    B = 15
    b = 10
    N = 100

    # Target choice:
    target = "junta"     # "random" or "junta"
    junta_S = 10          # only used if target="junta"

    num_seeds = 10
    accuracies = []

    for seed in range(num_seeds):
        print("\n" + "=" * 60)
        print(f"[Run] Seed {seed}")
        print("=" * 60)

        f_tt, ensemble, J = train_ensemble_column_only(
            B=B,
            b=b,
            N=N,
            seed=seed,
            tie_break_proj=1,
            run_espresso=False,
            verbose=False,
            target=target,
            junta_S=junta_S,
            relevant_set=None,
        )

        # 1) compute per-subcircuit accuracy + centered weights
        assign_centered_weights(f_tt, ensemble, B, center=0.5)

        # (optional) print a quick weight summary
        ws = [item["weight"] for item in ensemble]
        nonzero = sum(1 for w in ws if w > 0)
        print(f"[Weights] nonzero={nonzero}/{N}, "
              f"min={min(ws):.4f}, max={max(ws):.4f}, mean={statistics.mean(ws):.4f}")

        # 2) evaluate weighted ensemble
        acc = evaluate_ensemble_full_tt_weighted(
            f_tt=f_tt,
            ensemble=ensemble,
            B=B,
            tie_break_vote=1,
            verbose=False,
        )

        print(f"[Result] Seed {seed} weighted accuracy = {acc:.6f}")
        accuracies.append(acc)

    mean_acc = statistics.mean(accuracies)
    std_acc = statistics.stdev(accuracies) if num_seeds > 1 else 0.0

    print("\n" + "=" * 60)
    print("Final results over seeds (weighted vote)")
    print("=" * 60)
    print(f"B={B}, b={b}, N={N}, target={target}" + (f", S={junta_S}" if target == "junta" else ""))
    print(f"Mean accuracy: {mean_acc:.6f}")
    print(f"Std  accuracy: {std_acc:.6f}")


if __name__ == "__main__":
    main()
