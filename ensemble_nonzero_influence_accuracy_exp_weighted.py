# ensemble_column_nonzero_influence_accuracy_exp_weighted_vote.py
# -------------------------------------------------
# Train M b-bit projected subcircuits where columns are sampled ONLY from nonzero-influence bits,
# then aggregate at inference time using ACCURACY-weighted voting (Option 2: exponential cutoff).
#
# Training:
#   1) Build full truth table f on B bits (random or S-junta)
#   2) Compute exact influences Inf_i(f) for all i in [B]
#   3) Let NZ = { i : Inf_i(f) > eps_nonzero }
#   4) For each circuit m=1..M:
#        - sample b columns from NZ
#        - build projected truth table g_m on those b bits via majority marginalization
#        - (optional) run Espresso for stats only; always keep g_tt for fast inference
#
# Weighting (NEW):
#   5) Compute each subcircuit's full-table accuracy:
#        Acc_m = Pr_x [ g_m(proj_m(x)) == f(x) ]
#   6) Assign weights using exponential cutoff (Option 2):
#        centered_m = max(Acc_m - 0.5, 0)
#        raw_w_m = exp(gamma * centered_m) - 1
#        w_m = raw_w_m / sum_k raw_w_k
#      If all raw weights are 0 (all Acc<=0.5), fallback to uniform weights.
#
# Inference:
#   7) For each x in {0,1}^B:
#        y_hat = 1 if sum_m w_m * g_m(proj_m(x)) >= 0.5 * sum_m w_m else 0
#
# Reports:
#  - total time to compute influences over all B bits
#  - per-bit influences (and true junta bits if S-junta)
#  - nonzero-influence bits set
#  - total + average "espresso" training time for M circuits
#  - weight computation time (accuracy eval + weight assignment)
#  - accuracy over full 2^B truth table, total inference time, per-point inference time
#  - multi-seed mean/std of all the above (+ optional per-bit influence mean/std)

import time
import random
from typing import List, Dict, Tuple, Optional
import math


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
# Ensemble training (no weights yet)
# ==============================

def train_ensemble_unweighted(
    f_tt: List[int],
    B: int,
    M: int,
    b: int,
    influences: List[float],
    eps_nonzero: float = 0.0,
    use_espresso: bool = False,
) -> Tuple[List[Dict], float, float, List[int]]:
    """
    Train M subcircuits (store g_tt + cols). No aggregation weights yet.
    Returns (ensemble, total_train_time, avg_train_time, nz_bits).
    """
    nz_bits = nonzero_influence_bits(influences, eps=eps_nonzero)

    ensemble: List[Dict] = []
    total_train_time = 0.0

    for _ in range(M):
        cols = sample_cols_from_nonzero(B, b, nz_bits)
        g_tt = build_projected_tt_majority(f_tt, B, cols)

        t0 = time.time()
        model = train_espresso(g_tt, use_espresso=use_espresso)
        total_train_time += time.time() - t0

        ensemble.append({
            "cols": cols,
            "model": model,       # always contains g_tt
            "acc": None,          # filled later
            "weight": None,       # filled later
        })

    avg_train_time = total_train_time / M if M > 0 else 0.0
    return ensemble, total_train_time, avg_train_time, nz_bits


# ==============================
# Accuracy computation + weight assignment (Option 2)
# ==============================

def subcircuit_accuracy_full_tt(f_tt: List[int], B: int, cols: List[int], g_tt: List[int]) -> float:
    """
    Compute Acc = Pr_x [ g(proj(x)) == f(x) ] exactly by iterating over full truth table.
    """
    N = 1 << B
    correct = 0
    for x in range(N):
        u = proj_index(x, cols)
        if g_tt[u] == f_tt[x]:
            correct += 1
    return correct / N


def assign_exp_cutoff_weights(ensemble: List[Dict], gamma: float = 10.0) -> None:
    """
    Option 2: Exponential cutoff weights based on centered accuracy.
      centered = max(acc - 0.5, 0)
      raw_w = exp(gamma * centered) - 1
      weight = raw_w / sum(raw_w)
    If all raw weights are 0, fallback to uniform weights.
    """
    raws: List[float] = []
    for item in ensemble:
        acc = item["acc"]
        centered = acc - 0.5
        if centered < 0:
            centered = 0.0
        raw = math.exp(gamma * centered) - 1.0
        raws.append(raw)

    Z = sum(raws)
    if Z <= 0:
        # fallback: uniform
        M = len(ensemble)
        w = 1.0 / M if M > 0 else 0.0
        for item in ensemble:
            item["weight"] = w
        return

    for item, raw in zip(ensemble, raws):
        item["weight"] = raw / Z


def compute_accuracy_weights(
    f_tt: List[int],
    B: int,
    ensemble: List[Dict],
    gamma: float = 10.0,
) -> Tuple[float, float, float]:
    """
    1) compute each subcircuit accuracy on full truth table
    2) assign exp-cutoff weights (Option 2)
    Returns (total_time, acc_min, acc_max).
    """
    t0 = time.time()

    accs: List[float] = []
    for item in ensemble:
        cols = item["cols"]
        g_tt = item["model"]["g_tt"]
        acc = subcircuit_accuracy_full_tt(f_tt, B, cols, g_tt)
        item["acc"] = acc
        accs.append(acc)

    assign_exp_cutoff_weights(ensemble, gamma=gamma)

    total = time.time() - t0
    return total, (min(accs) if accs else 0.0), (max(accs) if accs else 0.0)


# ==============================
# Weighted inference + evaluation
# ==============================

def ensemble_predict_weighted(x: int, ensemble: List[Dict]) -> int:
    """
    Weighted vote:
      score = sum_m w_m * y_m
      predict 1 if score >= 0.5 * sum_m w_m else 0
    If sum weights is 0 (shouldn't happen with our fallback), use plain majority.
    """
    score = 0.0
    wsum = 0.0

    # weighted
    for item in ensemble:
        w = item["weight"]
        cols = item["cols"]
        g_tt = item["model"]["g_tt"]
        u = proj_index(x, cols)
        y = g_tt[u]
        score += w * y
        wsum += w

    if wsum <= 0:
        # fallback majority
        votes = 0
        for item in ensemble:
            cols = item["cols"]
            g_tt = item["model"]["g_tt"]
            u = proj_index(x, cols)
            votes += g_tt[u]
        return 1 if votes * 2 >= len(ensemble) else 0

    return 1 if score >= 0.5 * wsum else 0


def evaluate_ensemble_full_tt_weighted(f_tt: List[int], B: int, ensemble: List[Dict]) -> Tuple[float, float, float]:
    """
    Evaluate weighted ensemble over ALL 2^B points.
    Returns (accuracy, total_eval_time, per_point_time).
    """
    N = 1 << B
    t0 = time.time()

    correct = 0
    for x in range(N):
        y_hat = ensemble_predict_weighted(x, ensemble)
        if y_hat == f_tt[x]:
            correct += 1

    total = time.time() - t0
    acc = correct / N
    per_point = total / N if N > 0 else 0.0
    return acc, total, per_point


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
    gamma: float = 10.0,
    print_influences: bool = True,
) -> None:
    random.seed(seed)

    # Create f
    if S is None:
        f_tt = random_truth_table(B)
        print("Random Boolean function")
        junta_bits = None
    else:
        f_tt, junta_bits = random_s_junta_truth_table(B, S)
        print(f"S-junta function with S={S}")
        print(f"True junta bits: {junta_bits}")

    # influences
    influences, infl_time = compute_bit_influences(f_tt, B)
    if print_influences:
        print("\nBit influences:")
        for i, inf in enumerate(influences):
            print(f"  bit {i}: {inf:.6f}")
    print(f"Total time to compute influences: {infl_time:.6f} sec")

    nz_bits = nonzero_influence_bits(influences, eps=eps_nonzero)
    print(f"Nonzero-influence bits (eps={eps_nonzero}): {nz_bits}")

    # train subcircuits
    ensemble, total_train, avg_train, _ = train_ensemble_unweighted(
        f_tt=f_tt,
        B=B,
        M=M,
        b=b,
        influences=influences,
        eps_nonzero=eps_nonzero,
        use_espresso=use_espresso,
    )
    print("\nEspresso training time:")
    print(f"  Total:   {total_train:.6f} sec")
    print(f"  Average: {avg_train:.6f} sec per circuit")
    if use_espresso:
        used = sum(1 for it in ensemble if it["model"].get("espresso_used"))
        print(f"  Espresso used: {used}/{len(ensemble)} (depends on PyEDA availability)")
    else:
        print(f"  Espresso used: 0/{len(ensemble)} (fast inference via TT lookup)")

    # compute accuracy weights
    w_time, acc_min, acc_max = compute_accuracy_weights(f_tt, B, ensemble, gamma=gamma)
    weights = [it["weight"] for it in ensemble]
    print("\nAccuracy-weight computation:")
    print(f"  Total: {w_time:.6f} sec (includes full-table accuracy for each circuit)")
    print(f"  Acc min/max: {acc_min:.6f} / {acc_max:.6f}")
    print(f"  Weight min/mean/max: {min(weights):.6e} / {sum(weights)/len(weights):.6e} / {max(weights):.6e}")
    print(f"  gamma={gamma}")

    # evaluate weighted inference
    acc, eval_time, per_point = evaluate_ensemble_full_tt_weighted(f_tt, B, ensemble)
    print("\nInference evaluation (accuracy-weighted):")
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
    gamma: float = 10.0,
    print_influences_each_seed: bool = False,
    print_bit_influence_stats: bool = False,
) -> None:
    infl_times: List[float] = []
    train_totals: List[float] = []
    train_avgs: List[float] = []
    weight_times: List[float] = []
    accuracies: List[float] = []
    eval_times: List[float] = []
    per_point_times: List[float] = []
    all_bit_influences: List[List[float]] = []

    for seed in seeds:
        print(f"\n===== Seed {seed} =====")
        random.seed(seed)

        # Create f
        if S is None:
            f_tt = random_truth_table(B)
            print("Random Boolean function")
            junta_bits = None
        else:
            f_tt, junta_bits = random_s_junta_truth_table(B, S)
            print(f"S-junta function with S={S}")
            print(f"True junta bits: {junta_bits}")

        # influences
        influences, infl_time = compute_bit_influences(f_tt, B)
        infl_times.append(infl_time)
        all_bit_influences.append(influences)

        if print_influences_each_seed:
            print("Bit influences:")
            for i, inf in enumerate(influences):
                print(f"  bit {i}: {inf:.6f}")

        nz_bits = nonzero_influence_bits(influences, eps=eps_nonzero)
        print(f"Nonzero-influence bits (eps={eps_nonzero}): {nz_bits}")

        # train
        ensemble, total_train, avg_train, _ = train_ensemble_unweighted(
            f_tt=f_tt,
            B=B,
            M=M,
            b=b,
            influences=influences,
            eps_nonzero=eps_nonzero,
            use_espresso=use_espresso,
        )
        train_totals.append(total_train)
        train_avgs.append(avg_train)

        # compute accuracy weights
        w_time, acc_min, acc_max = compute_accuracy_weights(f_tt, B, ensemble, gamma=gamma)
        weight_times.append(w_time)

        # evaluate
        acc, eval_time, per_point = evaluate_ensemble_full_tt_weighted(f_tt, B, ensemble)
        accuracies.append(acc)
        eval_times.append(eval_time)
        per_point_times.append(per_point)

        print(f"Accuracy: {acc:.6f}")
        print(f"Inference total time: {eval_time:.6f} sec")
        print(f"Inference per-point time: {per_point:.12f} sec")
        print(f"Weighting time: {w_time:.6f} sec (Acc min/max: {acc_min:.3f}/{acc_max:.3f})")

    print("\n==============================")
    print("AVERAGED RESULTS OVER SEEDS")
    print("==============================")
    print(f"gamma={gamma}")

    m, s = mean_std(accuracies)
    print(f"Accuracy over full truth table: mean={m:.6f}, std={s:.6f}")

    m, s = mean_std(infl_times)
    print(f"Influence computation time: mean={m:.6f}, std={s:.6f}")

    m, s = mean_std(train_totals)
    print(f"Total Espresso training time: mean={m:.6f}, std={s:.6f}")

    m, s = mean_std(train_avgs)
    print(f"Average Espresso time per circuit: mean={m:.6f}, std={s:.6f}")

    m, s = mean_std(weight_times)
    print(f"Accuracy-weight computation time: mean={m:.6f}, std={s:.6f}")

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
    # S-junta on B=15 bits, train M circuits each on b=4 bits, sample cols only from nonzero-influence set,
    # and aggregate using accuracy-weighted vote (Option 2: exponential cutoff).
    run_multi_seed_experiment(
        B=15,
        M=50,
        b=4,
        S=8,
        seeds=list(range(10)),
        eps_nonzero=0.0,
        use_espresso=False,
        gamma=1.0,
        print_influences_each_seed=False,
        print_bit_influence_stats=False,
    )
