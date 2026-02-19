import time
import random
from typing import List, Dict, Tuple

# ==============================
# Utilities: truth table helpers
# ==============================

def random_truth_table(B: int) -> List[int]:
    """Uniform random Boolean function on B bits."""
    return [random.randint(0, 1) for _ in range(1 << B)]


def random_s_junta_truth_table(B: int, S: int) -> Tuple[List[int], List[int]]:
    """Random S‑junta function on B bits. Returns (tt, junta_bits)."""
    junta_bits = sorted(random.sample(range(B), S))

    # truth table for hidden function h on S bits
    h = [random.randint(0, 1) for _ in range(1 << S)]

    tt = []
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
    """Return influences for all bits and total computation time."""
    start = time.time()

    influences = []
    N = 1 << B

    for i in range(B):
        flips = 0
        mask = 1 << i
        for x in range(N):
            if tt[x] != tt[x ^ mask]:
                flips += 1
        influences.append(flips / N)

    total_time = time.time() - start
    return influences, total_time


# ==============================
# Projection utilities
# ==============================

def proj_index(x: int, cols: List[int]) -> int:
    """Project B‑bit integer x to index over selected columns."""
    idx = 0
    for j, bit in enumerate(cols):
        if (x >> bit) & 1:
            idx |= 1 << j
    return idx


# ==============================
# Majority projection to b bits
# ==============================

def build_projected_tt_majority(f_tt: List[int], B: int, cols: List[int]) -> List[int]:
    """Create b‑bit truth table via majority marginalization."""
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

    g_tt = [1 if counts1[i] >= counts0[i] else 0 for i in range(size)]
    return g_tt


# ==============================
# "Training" placeholder for Espresso
# ==============================

def train_espresso(g_tt: List[int]) -> Dict:
    """
    Placeholder for Espresso minimization.
    Replace with real call if PyEDA/Espresso available.
    """
    # Simulate small runtime cost
    return {"g_tt": g_tt}


# ==============================
# Ensemble training
# ==============================

def train_ensemble(
    f_tt: List[int],
    B: int,
    M: int,
    b: int,
    influences: List[float],
) -> Tuple[List[Dict], float, float]:
    """
    Train M circuits and return:
    - ensemble list
    - total espresso training time
    - average espresso training time
    """

    ensemble = []
    total_espresso_time = 0.0

    for _ in range(M):
        cols = sorted(random.sample(range(B), b))

        g_tt = build_projected_tt_majority(f_tt, B, cols)

        start = time.time()
        model = train_espresso(g_tt)
        elapsed = time.time() - start

        total_espresso_time += elapsed

        # influence weight = average global influence of selected bits
        weight = sum(influences[i] for i in cols) / b

        ensemble.append(
            {
                "cols": cols,
                "model": model,
                "weight": weight,
            }
        )

    avg_espresso_time = total_espresso_time / M if M > 0 else 0.0
    return ensemble, total_espresso_time, avg_espresso_time


# ==============================
# Inference
# ==============================

def predict_one(x: int, ensemble: List[Dict]) -> float:
    """Weighted probability prediction."""
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


# ==============================
# Accuracy over full truth table
# ==============================

def evaluate_accuracy(
    f_tt: List[int],
    B: int,
    ensemble: List[Dict],
) -> Tuple[float, float]:
    """Return accuracy and evaluation time over all 2^B inputs."""

    start = time.time()

    correct = 0
    for x in range(1 << B):
        prob = predict_one(x, ensemble)
        y_hat = 1 if prob >= 0.5 else 0
        if y_hat == f_tt[x]:
            correct += 1

    acc = correct / (1 << B)
    elapsed = time.time() - start

    return acc, elapsed


# ==============================
# Main experiment runner
# ==============================

def run_experiment(
    B: int,
    M: int,
    b: int,
    S: int = None,
    seed: int = 0,
):
    random.seed(seed)

    # ------------------------------
    # Generate function
    # ------------------------------
    if S is None:
        f_tt = random_truth_table(B)
        junta_bits = None
        print("Random Boolean function")
    else:
        f_tt, junta_bits = random_s_junta_truth_table(B, S)
        print(f"S‑junta function with S = {S}")

    # ------------------------------
    # Compute influences
    # ------------------------------
    influences, infl_time = compute_bit_influences(f_tt, B)

    print("\nBit influences:")
    for i, inf in enumerate(influences):
        print(f"  bit {i}: {inf:.6f}")

    print(f"Total time to compute influences: {infl_time:.4f} sec")

    if junta_bits is not None:
        print(f"True junta bits: {junta_bits}")

    # ------------------------------
    # Train ensemble
    # ------------------------------
    ensemble, total_train_time, avg_train_time = train_ensemble(
        f_tt, B, M, b, influences
    )

    print("\nEspresso training time:")
    print(f"  Total:   {total_train_time:.6f} sec")
    print(f"  Average: {avg_train_time:.6f} sec per circuit")

    # ------------------------------
    # Evaluate accuracy
    # ------------------------------
    acc, eval_time = evaluate_accuracy(f_tt, B, ensemble)

    print("\nInference evaluation:")
    print(f"  Accuracy over full truth table: {acc:.6f}")
    print(f"  Evaluation time: {eval_time:.6f} sec")


# ==============================
# Example run
# ==============================

def run_multi_seed_experiment(
    B: int,
    M: int,
    b: int,
    S: int = None,
    seeds: List[int] = list(range(10)),
):
    """Run experiment over multiple seeds and report mean/std of all metrics."""

    infl_times = []
    train_totals = []
    train_avgs = []
    accuracies = []
    eval_times = []

    all_bit_influences = []

    for seed in seeds:
        print(f"===== Seed {seed} =====")
        random.seed(seed)

        # Generate function
        if S is None:
            f_tt = random_truth_table(B)
            junta_bits = None
        else:
            f_tt, junta_bits = random_s_junta_truth_table(B, S)

        # Influences
        influences, infl_time = compute_bit_influences(f_tt, B)
        infl_times.append(infl_time)
        all_bit_influences.append(influences)

        # Train
        ensemble, total_train_time, avg_train_time = train_ensemble(
            f_tt, B, M, b, influences
        )
        train_totals.append(total_train_time)
        train_avgs.append(avg_train_time)

        # Evaluate
        acc, eval_time = evaluate_accuracy(f_tt, B, ensemble)
        accuracies.append(acc)
        eval_times.append(eval_time)

    # ==============================
    # Aggregate statistics
    # ==============================

    def mean_std(values: List[float]) -> Tuple[float, float]:
        mean = sum(values) / len(values)
        var = sum((v - mean) ** 2 for v in values) / len(values)
        return mean, var ** 0.5

    print("==============================")
    print("AVERAGED RESULTS OVER SEEDS")
    print("==============================")

    m, s = mean_std(eval_times)
    print(f"Inference evaluation time: mean={m:.6f}, std={s:.6f}")

    m, s = mean_std(train_avgs)
    print(f"Average Espresso time per circuit: mean={m:.6f}, std={s:.6f}\n")

    m, s = mean_std(accuracies)
    print(f"Accuracy over full truth table: mean={m:.6f}, std={s:.6f}")

    m, s = mean_std(infl_times)
    print(f"Influence computation time: mean={m:.8f}")

    m, s = mean_std(train_totals)
    print(f"Total Espresso training time: mean={m:.6f}")



    per_point_times = [t / (2 ** B) for t in eval_times]
    m, s = mean_std(per_point_times)
    print(f"Inference evaluation time per point: mean={m:.6f}")

    # # Bit influence mean/std per coordinate
    # print("Bit influence statistics across seeds:")
    # for i in range(B):
    #     vals = [inf_list[i] for inf_list in all_bit_influences]
    #     m, s = mean_std(vals)
    #     print(f"  bit {i}: mean={m:.6f}, std={s:.6f}")


# ==============================
# Example run
# ==============================

if __name__ == "__main__":
    run_multi_seed_experiment(
        B=15,   # total bits
        M=50,   # number of circuits
        b=4,    # bits per circuit
        S=8,    # set None for random function
        seeds=list(range(10)),
    )
