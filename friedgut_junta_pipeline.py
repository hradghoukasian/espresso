import numpy as np
import random
import time
from typing import List, Callable, Dict, Tuple, Optional

# ============================================================
# Utilities
# ============================================================

def proj_index(x: int, cols: List[int]) -> int:
    idx = 0
    for j, bit in enumerate(cols):
        if (x >> bit) & 1:
            idx |= 1 << j
    return idx


def random_truth_table(B: int) -> List[int]:
    return [random.randint(0, 1) for _ in range(1 << B)]


def random_s_junta_truth_table(B: int, S: int) -> Tuple[List[int], List[int]]:
    junta_bits = sorted(random.sample(range(B), S))
    h = [random.randint(0, 1) for _ in range(1 << S)]

    tt = []
    for x in range(1 << B):
        idx = 0
        for j, bit in enumerate(junta_bits):
            if (x >> bit) & 1:
                idx |= 1 << j
        tt.append(h[idx])

    return tt, junta_bits


# ============================================================
# STEP 1 — Influence estimation
# ============================================================

def influences_exact_tt(f_tt: List[int], B: int) -> List[float]:
    N = 1 << B
    infl = []
    for i in range(B):
        flips = 0
        mask = 1 << i
        for x in range(N):
            if f_tt[x] != f_tt[x ^ mask]:
                flips += 1
        infl.append(flips / N)
    return infl


def influences_oracle_uniform(
    oracle: Callable[[int], int], B: int, T: int
) -> List[float]:
    infl = [0.0] * B
    for i in range(B):
        flips = 0
        for _ in range(T):
            x = random.randrange(1 << B)
            if oracle(x) != oracle(x ^ (1 << i)):
                flips += 1
        infl[i] = flips / T
    return infl


def influences_from_dataset(
    data: List[int], oracle: Callable[[int], int], B: int
) -> List[float]:
    """
    Uses only observed pairs inside dataset.
    Hoeffding applies to #matched pairs per bit.
    """
    data_set = set(data)
    infl = [0.0] * B

    for i in range(B):
        pairs = 0
        flips = 0
        mask = 1 << i
        for x in data:
            y = x ^ mask
            if y in data_set:
                pairs += 1
                if oracle(x) != oracle(y):
                    flips += 1
        infl[i] = flips / pairs if pairs > 0 else 0.0

    return infl


# ============================================================
# STEP 2 — Select influential set
# ============================================================

def select_J_tau_K(infl: List[float], tau: float, K: int) -> List[int]:
    J = [i for i, v in enumerate(infl) if v > tau]
    random.shuffle(J)
    J.sort(key=lambda i: infl[i], reverse=True)
    return sorted(J[:K])


# ============================================================
# STEP 3 — Build surrogate f_tau
# ============================================================

def f_tau_exact_marginal(
    f_tt: List[int], B: int, J: List[int]
) -> List[int]:
    K = len(J)
    size = 1 << K
    counts0 = [0] * size
    counts1 = [0] * size

    for x in range(1 << B):
        u = proj_index(x, J)
        if f_tt[x] == 0:
            counts0[u] += 1
        else:
            counts1[u] += 1

    return [1 if counts1[u] >= counts0[u] else 0 for u in range(size)]


def f_tau_monte_carlo(
    oracle: Callable[[int], int], B: int, J: List[int], R: int
) -> List[int]:
    K = len(J)
    size = 1 << K
    out = []

    for u in range(size):
        ones = 0
        for _ in range(R):
            z = random.randrange(1 << (B - K))
            x = 0

            # fill J bits
            for j, bit in enumerate(J):
                if (u >> j) & 1:
                    x |= 1 << bit

            # fill complement bits
            zi = 0
            for bit in range(B):
                if bit not in J:
                    if (z >> zi) & 1:
                        x |= 1 << bit
                    zi += 1

            ones += oracle(x)

        out.append(1 if ones >= R / 2 else 0)

    return out


# -------- FWHT ----------

def fwht(a):
    h = 1
    n = len(a)
    a = a.copy()
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(i, i + h):
                x = a[j]
                y = a[j + h]
                a[j] = x + y
                a[j + h] = x - y
        h *= 2
    return a


def f_tau_fwht(f_tt: List[int], B: int, J: List[int]) -> List[int]:
    arr = np.array([1 if v else -1 for v in f_tt], dtype=float)
    F = fwht(arr)

    for S in range(len(F)):
        for i in range(B):
            if ((S >> i) & 1) and (i not in J):
                F[S] = 0
                break

    h = fwht(F) / (1 << B)

    K = len(J)
    size = 1 << K
    out = []

    for u in range(size):
        x = 0
        for j, bit in enumerate(J):
            if (u >> j) & 1:
                x |= 1 << bit
        out.append(1 if h[x] >= 0 else 0)

    return out


# ============================================================
# STEP 4 — Espresso learning (optional)
# ============================================================

def learn_espresso(f_tau_tt: List[int]):
    try:
        from pyeda.boolalg.espresso import espresso_tts
        from pyeda.boolalg.bfarray import exprvars
        from pyeda.boolalg.tt import truthtable
    except Exception:
        return {"expr": None, "g_tt": f_tau_tt}

    K = (len(f_tau_tt)).bit_length() - 1
    X = exprvars("x", K)
    outs = "".join("1" if v else "0" for v in f_tau_tt)
    tt = truthtable(X, outs)
    minimized = espresso_tts(tt)

    return {"expr": minimized[0] if minimized else None, "g_tt": f_tau_tt}


# ============================================================
# Accuracy
# ============================================================

def accuracy_full(f_tt: List[int], B: int, model: Dict, J: List[int]) -> float:
    correct = 0
    for x in range(1 << B):
        u = proj_index(x, J)
        y_hat = model["g_tt"][u]
        if y_hat == f_tt[x]:
            correct += 1
    return correct / (1 << B)


# ============================================================
# Main pipeline
# ============================================================

def run_pipeline(
    seeds=10,
    B=12,
    S=6,
    tau=0.01,
    K=4,
    T=500,
    R=200,
    step1="exact",
    step3="exact",
    use_espresso=False,
):
    if isinstance(seeds, int):
        seeds = list(range(seeds))

    accs = []

    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed)

        # --- same logic as run_pipeline but silent ---
        f_tt, _ = random_s_junta_truth_table(B, S)
        oracle = lambda x: f_tt[x]

        # Step 1
        if step1 == "exact":
            infl = influences_exact_tt(f_tt, B)
        elif step1 == "oracle":
            infl = influences_oracle_uniform(oracle, B, T)
        else:
            raise ValueError("Unknown step1")

        # Step 2
        J = select_J_tau_K(infl, tau, K)

        # Step 3
        if step3 == "exact":
            f_tau_tt = f_tau_exact_marginal(f_tt, B, J)
        elif step3 == "mc":
            f_tau_tt = f_tau_monte_carlo(oracle, B, J, R)
        elif step3 == "fwht":
            f_tau_tt = f_tau_fwht(f_tt, B, J)
        else:
            raise ValueError("Unknown step3")

        # Step 4
        model = learn_espresso(f_tau_tt) if use_espresso else {"g_tt": f_tau_tt}

        # Accuracy
        acc = accuracy_full(f_tt, B, model, J)
        accs.append(acc)

    mean_acc = float(np.mean(accs))

    print("\n=== MULTI-SEED RESULT ===")
    print(f"Seeds: {len(seeds)}")
    print(f"Average accuracy: {mean_acc:.6f}")


# ============================================================
# Run experiment
# ============================================================

if __name__ == "__main__":
    run_pipeline(
        seeds=10,          # number of random seeds
        B=12,
        S=8,
        tau=0.01,
        K=4,
        T=1000,
        R=200,
        step1="exact",     # exact | oracle
        step3="fwht",     # exact | mc | fwht
        use_espresso=False,
    )

