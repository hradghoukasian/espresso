# ensemble_column_nonzero_influence_weighted_vote.py
# -------------------------------------------------
# Influence-weighted ensemble of b-bit projected truth tables,
# where each circuit's b columns are sampled ONLY from bits with nonzero influence.
#
# Fast inference: per-circuit truth-table lookup (no Espresso eval at inference).
# Optional: attempt Espresso minimization for stats only (keeps g_tt regardless).
#
# Reports:
#  - total time to compute influences over all B bits
#  - per-bit influences (and true junta bits if S-junta)
#  - nonzero-influence bits set
#  - total + average "espresso" training time for M circuits
#  - accuracy over full 2^B truth table, total inference time, per-point inference time
#  - multi-seed mean/std of all the above (+ optional per-bit influence mean/std)

import time
import random
from typing import List, Dict, Tuple, Optional
import math


# ==============================
# softmax helper function
# ==============================
def softmax_weights(scores: List[float], tau: float = 1.0) -> List[float]:
    """
    Stable softmax with temperature tau > 0:
        w_m = exp(score_m / tau) / sum_k exp(score_k / tau)

    larger tau -> more peaky (more weight on largest score).
    smaller tau -> closer to uniform.
    """
    if tau <= 0:
        raise ValueError("tau must be > 0 for softmax temperature.")

    scaled = [s * tau for s in scores]
    m = max(scaled)  # stability
    exps = [math.exp(z-m) for z in scaled]
    Z = sum(exps)
    return [e / Z for e in exps]


# ==============================
# Truth table generators
# ==============================
def random_truth_table(B: int) -> List[int]:
    """Uniform random Boolean function on B bits."""
    return [random.randint(0, 1) for _ in range(1 << B)]


def random_s_junta_truth_table(B: int, S: int) -> Tuple[List[int], List[int]]:
    """
    Random S-junta on B bits:
      - choose S relevant coordinates J
      - choose random h: {0,1}^S -> {0,1}
      - define f(x) = h(x_J)
    Returns (truth_table, junta_bits).
    """
    junta_bits = sorted(random.sample(range(B), S))
    h = [random.randint(0, 1) for _ in range(1 << S)]

    tt: List[int] = []
    for x in range(1 << B):
        idx = 0
        for j, bit in enumerate(junta_bits):
            if (x >> bit) & 1:
                idx |= 1 << j
        tt.append(h[idx])

    return tt, junta_bits


# ==============================
# Influence computation
# ==============================
def compute_bit_influences(tt: List[int], B: int) -> Tuple[List[float], float]:
    """
    Compute Inf_i(f) exactly from full truth table under uniform measure:
      Inf_i = Pr_x[f(x) != f(x^i)]
    Returns (influences, total_time_seconds).
    """
    start = time.time()
    N = 1 << B
    influences: List[float] = []

    for i in range(B):
        flips = 0
        mask = 1 << i
        for x in range(N):
            if tt[x] != tt[x ^ mask]:
                flips += 1
        influences.append(flips / N)

    return influences, time.time() - start


def nonzero_influence_bits(influences: List[float], eps: float = 0.0) -> List[int]:
    """Return indices i with Inf_i > eps."""
    return [i for i, inf in enumerate(influences) if inf > eps]


# ==============================
# Projection utilities
# ==============================
def proj_index(x: int, cols: List[int]) -> int:
    """Project B-bit integer x onto cols -> b-bit integer index."""
    idx = 0
    for j, bit in enumerate(cols):
        if (x >> bit) & 1:
            idx |= 1 << j
    return idx


def build_projected_tt_majority(f_tt: List[int], B: int, cols: List[int]) -> List[int]:
    """
    Build b-bit truth table g(u) by majority marginalization:
      g(u) = majority_{x: proj(x,cols)=u} f(x)
    Tie-break: 1 if counts equal.
    """
    b = len(cols)
    size = 1 << b
    counts0 = [0] * size
    counts1 = [0] * size

    for x in range(1 << B):
        u = proj_index(x, cols)
        if f_tt[x] == 0:
            counts0[u] += 1
        else:
            counts1[u] += 1

    return [1 if counts1[u] >= counts0[u] else 0 for u in range(size)]


# ==============================
# Optional Espresso minimization (stats only)
# ==============================
def _try_import_pyeda():
    try:
        from pyeda.boolalg.espresso import espresso_tts  # type: ignore
        from pyeda.boolalg.bfarray import exprvars      # type: ignore
        from pyeda.boolalg.tt import truthtable         # type: ignore
        return espresso_tts, exprvars, truthtable
    except Exception:
        return None


def train_espresso(g_tt: List[int], use_espresso: bool = False) -> Dict:
    """
    Always stores g_tt for fastest inference.
    If use_espresso and PyEDA is available, also computes an Espresso-minimized form (expr) for stats.
    """
    model = {"g_tt": g_tt, "expr": None, "espresso_used": False}

    if not use_espresso:
        return model

    deps = _try_import_pyeda()
    if deps is None:
        return model

    espresso_tts, exprvars, truthtable = deps
    b = (len(g_tt)).bit_length() - 1

    outs = "".join("1" if v else "0" for v in g_tt)
    X = exprvars("x", b)
    tt = truthtable(X, outs)

    minimized = espresso_tts(tt)
    model["expr"] = minimized[0] if minimized else None
    model["espresso_used"] = True
    return model


# ==============================
# Smart nonzero-influence sampling
# ==============================
def sample_cols_from_nonzero(B: int, b: int, nz_bits: List[int]) -> List[int]:
    """
    Sample b distinct columns from nz_bits if possible.
    If nz_bits has < b elements, fill remaining columns from the complement of nz_bits.
    """
    if len(nz_bits) >= b:
        return sorted(random.sample(nz_bits, b))

    remaining = [i for i in range(B) if i not in nz_bits]
    fill = random.sample(remaining, b - len(nz_bits))
    return sorted(nz_bits + fill)


# ==============================
# Ensemble training / inference
# ==============================
def train_ensemble(
    f_tt: List[int],
    B: int,
    M: int,
    b: int,
    influences: List[float],
    eps_nonzero: float = 0.0,
    use_espresso: bool = False,
    tau: float = 1.0,  # temperature for softmax over circuit influence scores
) -> Tuple[List[Dict], float, float, List[int]]:
    """
    Train M circuits:
      - choose cols from nonzero-influence bits
      - build projected g_tt via majority
      - (optional) run espresso for stats
      - compute circuit score = average global influence of selected cols
      - set final circuit weights via softmax(scores, tau)

    Returns:
      (ensemble, total_train_time, avg_train_time, nz_bits)
    """
    nz_bits = nonzero_influence_bits(influences, eps=eps_nonzero)

    ensemble: List[Dict] = []
    scores: List[float] = []
    total_train_time = 0.0

    for _ in range(M):
        cols = sample_cols_from_nonzero(B, b, nz_bits)
        g_tt = build_projected_tt_majority(f_tt, B, cols)

        t0 = time.time()
        model = train_espresso(g_tt, use_espresso=use_espresso)
        total_train_time += time.time() - t0

        score = sum(influences[i] for i in cols) / b
        scores.append(score)

        ensemble.append({
            "cols": cols,
            "model": model,   # always has g_tt
            "weight": None,   # filled after softmax
            "score": score,   # keep for debugging/analysis if you want
        })

    # softmax over circuit scores -> final weights
    weights = softmax_weights(scores, tau=tau)
    for item, w in zip(ensemble, weights):
        item["weight"] = w

    avg_train_time = total_train_time / M if M > 0 else 0.0
    return ensemble, total_train_time, avg_train_time, nz_bits


def predict_one(x: int, ensemble: List[Dict]) -> float:
    """Weighted probability prediction using fast g_tt lookup."""
    num = 0.0
    den = 0.0
    for item in ensemble:
        cols = item["cols"]
        g_tt = item["model"]["g_tt"]
        w = item["weight"]

        u = proj_index(x, cols)
        y = g_tt[u]

        num += w * y
        den += w

    return num / den if den > 0 else 0.0


def evaluate_accuracy(f_tt: List[int], B: int, ensemble: List[Dict]) -> Tuple[float, float, float]:
    """
    Evaluate over ALL 2^B points.
    Returns (accuracy, total_eval_time, per_point_time).
    """
    N = 1 << B
    t0 = time.time()

    correct = 0
    for x in range(N):
        p = predict_one(x, ensemble)
        y_hat = 1 if p >= 0.5 else 0
        if y_hat == f_tt[x]:
            correct += 1

    total_time = time.time() - t0
    acc = correct / N
    per_point = total_time / N if N > 0 else 0.0
    return acc, total_time, per_point


# ==============================
# Reporting helpers
# ==============================
def mean_std(vals: List[float]) -> Tuple[float, float]:
    mu = sum(vals) / len(vals)
    var = sum((v - mu) ** 2 for v in vals) / len(vals)
    return mu, var ** 0.5


# ==============================
# Experiment runners
# ==============================
def run_experiment(
    B: int,
    M: int,
    b: int,
    S: Optional[int] = None,
    seed: int = 0,
    eps_nonzero: float = 0.0,
    use_espresso: bool = False,
    tau: float = 1.0,
    print_influences: bool = True,
) -> None:
    random.seed(seed)

    # Create f
    if S is None:
        f_tt = random_truth_table(B)
        print("Random Boolean function")
    else:
        f_tt, junta_bits = random_s_junta_truth_table(B, S)
        print(f"S-junta function with S={S}")
        print(f"True junta bits: {junta_bits}")

    # Compute influences
    influences, infl_time = compute_bit_influences(f_tt, B)

    if print_influences:
        print("\nBit influences:")
        for i, inf in enumerate(influences):
            print(f"  bit {i}: {inf:.6f}")
    print(f"Total time to compute influences: {infl_time:.6f} sec")

    nz_bits = nonzero_influence_bits(influences, eps=eps_nonzero)
    print(f"Nonzero-influence bits (eps={eps_nonzero}): {nz_bits}")

    # Train ensemble
    ensemble, total_train, avg_train, _ = train_ensemble(
        f_tt=f_tt,
        B=B,
        M=M,
        b=b,
        influences=influences,
        eps_nonzero=eps_nonzero,
        use_espresso=use_espresso,
        tau=tau,
    )

    print("\nEspresso training time:")
    print(f"  Total:   {total_train:.6f} sec")
    print(f"  Average: {avg_train:.6f} sec per circuit")
    if use_espresso:
        used = sum(1 for it in ensemble if it["model"].get("espresso_used"))
        print(f"  Espresso used: {used}/{len(ensemble)} (depends on PyEDA availability)")
    else:
        print(f"  Espresso used: 0/{len(ensemble)} (fast inference via TT lookup)")

    # Debug (optional): show score/weight range
    scores = [it["score"] for it in ensemble]
    weights = [it["weight"] for it in ensemble]
    print(f"\nSoftmax tau={tau}")
    print(f"  circuit score min/mean/max: {min(scores):.6f} / {sum(scores)/len(scores):.6f} / {max(scores):.6f}")
    print(f"  weight        min/mean/max: {min(weights):.6e} / {sum(weights)/len(weights):.6e} / {max(weights):.6e}")

    # Evaluate inference
    acc, eval_time, per_point = evaluate_accuracy(f_tt, B, ensemble)
    print("\nInference evaluation:")
    print(f"  Accuracy over full truth table: {acc:.6f}")
    print(f"  Total evaluation time: {eval_time:.6f} sec")
    print(f"  Inference time per point: {per_point:.12f} sec")


def run_multi_seed_experiment(
    B: int,
    M: int,
    b: int,
    S: Optional[int] = None,
    seeds: List[int] = list(range(10)),
    eps_nonzero: float = 0.0,
    use_espresso: bool = False,
    tau: float = 1.0,
    print_influences_each_seed: bool = False,
    print_bit_influence_stats: bool = False,
) -> None:
    infl_times: List[float] = []
    train_totals: List[float] = []
    train_avgs: List[float] = []
    accuracies: List[float] = []
    eval_times: List[float] = []
    per_point_times: List[float] = []
    all_bit_influences: List[List[float]] = []

    for seed in seeds:
        print(f"\n===== Seed {seed} =====")
        random.seed(seed)

        if S is None:
            f_tt = random_truth_table(B)
            print("Random Boolean function")
        else:
            f_tt, junta_bits = random_s_junta_truth_table(B, S)
            print(f"S-junta function with S={S}")
            print(f"True junta bits: {junta_bits}")

        influences, infl_time = compute_bit_influences(f_tt, B)
        infl_times.append(infl_time)
        all_bit_influences.append(influences)

        if print_influences_each_seed:
            print("Bit influences:")
            for i, inf in enumerate(influences):
                print(f"  bit {i}: {inf:.6f}")

        nz_bits = nonzero_influence_bits(influences, eps=eps_nonzero)
        print(f"Nonzero-influence bits (eps={eps_nonzero}): {nz_bits}")

        ensemble, total_train, avg_train, _ = train_ensemble(
            f_tt=f_tt,
            B=B,
            M=M,
            b=b,
            influences=influences,
            eps_nonzero=eps_nonzero,
            use_espresso=use_espresso,
            tau=tau,
        )
        train_totals.append(total_train)
        train_avgs.append(avg_train)

        acc, eval_time, per_point = evaluate_accuracy(f_tt, B, ensemble)
        accuracies.append(acc)
        eval_times.append(eval_time)
        per_point_times.append(per_point)

        print(f"Accuracy: {acc:.6f}")
        print(f"Inference total time: {eval_time:.6f} sec")
        print(f"Inference per-point time: {per_point:.12f} sec")

    print("\n==============================")
    print("AVERAGED RESULTS OVER SEEDS")
    print("==============================")
    print(f"Softmax tau={tau}")

    m, s = mean_std(accuracies)
    print(f"Accuracy over full truth table: mean={m:.6f}, std={s:.6f}")

    m, s = mean_std(infl_times)
    print(f"Influence computation time: mean={m:.6f}, std={s:.6f}")

    m, s = mean_std(train_totals)
    print(f"Total Espresso training time: mean={m:.6f}, std={s:.6f}")

    m, s = mean_std(train_avgs)
    print(f"Average Espresso time per circuit: mean={m:.6f}, std={s:.6f}")

    m, s = mean_std(eval_times)
    print(f"Inference total evaluation time: mean={m:.6f}, std={s:.6f}")

    m, s = mean_std(per_point_times)
    print(f"Inference time per point: mean={m:.12f}, std={s:.12f}")

    if print_bit_influence_stats:
        print("\nBit influence statistics across seeds:")
        for i in range(B):
            vals = [infl[i] for infl in all_bit_influences]
            mu, sd = mean_std(vals)
            print(f"  bit {i}: mean={mu:.6f}, std={sd:.6f}")


if __name__ == "__main__":
    # Example:
    # 8-junta on B=15 bits, M=50 circuits each on b=4 bits,
    # choose circuit bits from nonzero-influence set.
    # Tune tau
    run_multi_seed_experiment(
        B=15,
        M=50,
        b=4,
        S=8,
        seeds=list(range(10)),
        eps_nonzero=0.0,
        use_espresso=False,
        tau=100,
        print_influences_each_seed=False,
        print_bit_influence_stats=False,
    )