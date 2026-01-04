# ensemble_column_unweighted_vote.py
import random
import math
from typing import List, Dict, Optional, Tuple
import statistics

# Only needed if we set run_espresso=True
from espresso_minimization import minimize_truth_table_espresso


# ----------------------------
# Target function generators
# ----------------------------

def random_truth_table(B: int, seed: int = 0) -> List[int]:
    """
    Random full truth table f:{0,1}^B -> {0,1}, in lexicographic order
    indexed by integer x in [0, 2^B).
    """
    rng = random.Random(seed)
    size = 1 << B
    return [rng.getrandbits(1) for _ in range(size)]


def random_s_junta_truth_table(
    B: int,
    S: int,
    seed: int = 0,
    relevant_set: Optional[List[int]] = None,
) -> Tuple[List[int], List[int]]:
    """
    Construct an S-junta f:{0,1}^B->{0,1}:
        pick relevant bits J of size S
        pick a random Boolean function h:{0,1}^S->{0,1}
        define f(x)=h(x[J])

    Returns:
        f_tt: length 2^B list of 0/1
        J: list of relevant bit indices (0-indexed, length S)
    """
    if not (0 <= S <= B):
        raise ValueError("Require 0 <= S <= B")

    rng = random.Random(seed)

    # Choose relevant set J
    if relevant_set is None:
        J = sorted(rng.sample(range(B), S))
    else:
        if len(relevant_set) != S:
            raise ValueError("relevant_set must have length S")
        if len(set(relevant_set)) != S or any(j < 0 or j >= B for j in relevant_set):
            raise ValueError("relevant_set must be S distinct indices in [0,B)")
        J = sorted(relevant_set)

    # Random truth table for h on S bits
    h_size = 1 << S
    h_tt = [rng.getrandbits(1) for _ in range(h_size)]

    # Build f truth table over B bits
    size_B = 1 << B
    f_tt = [0] * size_B

    for x in range(size_B):
        # compute u = projection of x onto J, in the same order as J
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
    """
    Project B-bit integer x onto the bits at positions in cols (0-indexed),
    producing a b-bit integer u in [0, 2^b).
    The k-th bit of u corresponds to cols[k].
    """
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
    """
    Given the *full* truth table f_tt over B bits, build the projected truth table
    g over b=|cols| bits by majority vote over marginalized variables.

    For each u in {0,1}^b:
        g(u) = majority{ f(x) : proj(x)=u }  (ties resolved by tie_break)
    """
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
    """
    Probability that a uniformly random b-subset of [B] contains a fixed S-subset J.
    p = C(B-S, b-S) / C(B, b)  (for b>=S), else 0.
    """
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
    # target controls:
    target: str = "random",          # "random" or "junta"
    junta_S: Optional[int] = None,   # used iff target="junta"
    relevant_set: Optional[List[int]] = None,
) -> Tuple[List[int], List[Dict], Optional[List[int]]]:
    """
    Train N subcircuits on random b-column subsets, without row sampling.

    Args:
        B: total input bits
        b: bits per subcircuit (columns)
        N: number of subcircuits
        seed: RNG seed
        tie_break_proj: tie-break when building projected truth table
        run_espresso: if True, run Espresso minimization per subcircuit
        verbose: print progress
        target: "random" (random f on B bits) or "junta" (S-junta on B bits)
        junta_S: S (required if target="junta")
        relevant_set: optionally fix the junta relevant indices

    Returns:
        f_tt: length 2^B list of 0/1
        ensemble: list of dicts with keys {"cols","g_tt","espresso_stats","overlap"}
        J: relevant set (if junta), else None
    """
    if not (0 < b <= B):
        raise ValueError("Require 0 < b <= B")

    # Build target truth table
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
# Inference + evaluation
# ----------------------------

def ensemble_predict_majority(
    x: int,
    ensemble: List[Dict],
    tie_break_vote: int = 1,
) -> int:
    """
    Majority vote across subcircuits.
    Each subcircuit output is g_n(x[S_n]) obtained by indexing g_tt at projected u.
    """
    N = len(ensemble)
    votes = 0
    for item in ensemble:
        u = proj_index(x, item["cols"])
        votes += 1 if item["g_tt"][u] == "1" else 0

    if 2 * votes > N:
        return 1
    if 2 * votes < N:
        return 0
    return 1 if tie_break_vote == 1 else 0


def evaluate_ensemble_full_tt(
    f_tt: List[int],
    ensemble: List[Dict],
    B: int,
    tie_break_vote: int = 1,
    verbose: bool = True,
    print_every: int = 200000,
) -> float:
    """
    Compute accuracy over the *entire* truth table of size 2^B.
    """
    size_B = 1 << B
    correct = 0
    for x in range(size_B):
        if verbose and (x % print_every == 0) and x > 0:
            print(f"[Eval] {x}/{size_B} ...")

        y_hat = ensemble_predict_majority(x, ensemble, tie_break_vote=tie_break_vote)
        if y_hat == f_tt[x]:
            correct += 1

    if verbose:
        print("[Eval] done.")
    return correct / size_B


def main():
    # ---- Experiment params ----
    B = 15
    b = 10
    N = 100

    # Target choice:
    target = "junta"     # "random" or "junta"
    junta_S = 10        # only used if target="junta"

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
            run_espresso=False,   # turn on for minimization stats
            verbose=False,
            target=target,
            junta_S=junta_S,
            relevant_set=None,
        )

        acc = evaluate_ensemble_full_tt(
            f_tt=f_tt,
            ensemble=ensemble,
            B=B,
            tie_break_vote=1,
            verbose=False,        # suppress per-x evaluation logs
        )

        print(f"[Result] Seed {seed} accuracy = {acc:.6f}")
        accuracies.append(acc)

    mean_acc = statistics.mean(accuracies)
    std_acc = statistics.stdev(accuracies) if num_seeds > 1 else 0.0

    print("\n" + "=" * 60)
    print("Final results over seeds")
    print("=" * 60)
    print(f"B={B}, b={b}, N={N}, target={target}" + (f", S={junta_S}" if target == "junta" else ""))
    print(f"Mean accuracy: {mean_acc:.6f}")
    print(f"Std  accuracy: {std_acc:.6f}")

if __name__ == "__main__":
    main()
