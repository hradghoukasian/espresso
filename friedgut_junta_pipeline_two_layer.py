import numpy as np
import random
from typing import List, Callable, Dict, Tuple, Optional


# ============================================================
# Utilities
# ============================================================

def proj_index(x: int, cols: List[int]) -> int:
    """Project an integer bitstring x onto coordinates in cols, packed into [0, 2^|cols|)."""
    idx = 0
    for j, bit in enumerate(cols):
        if (x >> bit) & 1:
            idx |= 1 << j
    return idx


def random_truth_table(B: int) -> List[int]:
    """Uniform random Boolean function f:{0,1}^B -> {0,1}, returned as length-2^B truth table."""
    return [random.randint(0, 1) for _ in range(1 << B)]


def ones_fraction(tt: List[int]) -> float:
    return sum(tt) / len(tt)

def random_s_junta_truth_table(B: int, S: int) -> Tuple[List[int], List[int]]:
    """
    Random S-junta on B bits:
      - choose S relevant bits uniformly
      - choose a random truth table h:{0,1}^S -> {0,1}
      - define f(x) = h(x restricted to relevant bits)
    Returns (f_tt, junta_bits).
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


# ============================================================
# STEP 1 — Influence estimation  Inf_b^T
# ============================================================

def influences_exact_tt(f_tt: List[int], B: int) -> List[float]:
    """Exact influence under uniform measure using the full truth table."""
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


def influences_oracle_uniform(oracle: Callable[[int], int], B: int, T: int) -> List[float]:
    """Empirical influence using T uniform random samples per coordinate."""
    infl = [0.0] * B
    for i in range(B):
        flips = 0
        mask = 1 << i
        for _ in range(T):
            x = random.randrange(1 << B)
            if oracle(x) != oracle(x ^ mask):
                flips += 1
        infl[i] = flips / T
    return infl


def influences_from_dataset(data: List[int], oracle: Callable[[int], int], B: int) -> List[float]:
    """
    Uses only observed pairs inside dataset (x and x^e_i both present).
    Hoeffding would apply to the #matched pairs per bit.
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
# STEP 2 — Select influential set J_{tau,K}
# ============================================================

def select_J_tau_K(infl: List[float], tau: float, K: int) -> List[int]:
    """
    J_tau := {i : Inf_i > tau}
    Then pick top-K by Inf_i, breaking ties randomly.
    """
    J = [i for i, v in enumerate(infl) if v > tau]
    random.shuffle(J)  # random tie-break
    J.sort(key=lambda i: infl[i], reverse=True)
    return sorted(J[:K])


# ============================================================
# STEP 3 — Build surrogate f_tau on J_{tau,K}
# ============================================================

def f_tau_exact_marginal(f_tt: List[int], B: int, J: List[int]) -> List[int]:
    """
    Exact marginal majority:
      f_tau(u) := 1{ E_z[f(u,z)] >= 1/2 }
    computed by counting over all x with the same projection u.
    Returns truth table over {0,1}^{|J|} (length 2^|J|).
    """
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


def f_tau_monte_carlo(oracle: Callable[[int], int], B: int, J: List[int], R: int) -> List[int]:
    """
    Monte Carlo estimate of the marginal majority for each u in {0,1}^{|J|},
    using R random completions of the complement bits.
    """
    K = len(J)
    size = 1 << K
    out: List[int] = []

    # Precompute complement bits (order matters for packing z)
    J_set = set(J)
    comp_bits = [b for b in range(B) if b not in J_set]
    assert len(comp_bits) == B - K

    for u in range(size):
        ones = 0
        for _ in range(R):
            z = random.randrange(1 << (B - K))
            x = 0

            # fill J bits from u
            for j, bit in enumerate(J):
                if (u >> j) & 1:
                    x |= 1 << bit

            # fill complement bits from z
            for zi, bit in enumerate(comp_bits):
                if (z >> zi) & 1:
                    x |= 1 << bit

            ones += oracle(x)

        out.append(1 if ones >= R / 2 else 0)

    return out


# ----- FWHT (optional Step 3 variant) ----------

def fwht(a: np.ndarray) -> np.ndarray:
    h = 1
    n = len(a)
    out = a.copy()
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(i, i + h):
                x = out[j]
                y = out[j + h]
                out[j] = x + y
                out[j + h] = x - y
        h *= 2
    return out


def f_tau_fwht(f_tt: List[int], B: int, J: List[int]) -> List[int]:
    """
    Fourier truncation to subsets contained in J, then sign to get a Boolean surrogate.
    (This is a different surrogate than the exact marginal majority; keep if you want.)
    """
    arr = np.array([1.0 if v else -1.0 for v in f_tt], dtype=float)
    F = fwht(arr)

    J_set = set(J)
    for S in range(len(F)):
        # If S contains a bit not in J, zero it out
        for i in range(B):
            if ((S >> i) & 1) and (i not in J_set):
                F[S] = 0.0
                break

    h = fwht(F) / float(1 << B)

    K = len(J)
    size = 1 << K
    out: List[int] = []

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

def learn_espresso(f_tau_tt: List[int]) -> Dict:
    """
    If pyeda is installed, runs Espresso minimization on f_tau_tt.
    Always returns a dict with:
      - "expr": minimized expression or None
      - "g_tt": the surrogate truth table over {0,1}^{|J|}
    """
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
# Prediction helpers (needed for residual boosting)
# ============================================================

def predict_stage_on_x(x: int, model: Dict, J: List[int]) -> int:
    """Predict using a stage model (truth table on J) for a single x in {0,1}^B."""
    u = proj_index(x, J)
    return int(model["g_tt"][u])


def predict_stage_full_tt(B: int, model: Dict, J: List[int]) -> List[int]:
    """Predict a stage model on all x in {0,1}^B, returning a length-2^B truth table."""
    return [predict_stage_on_x(x, model, J) for x in range(1 << B)]


def xor_tt(a_tt: List[int], b_tt: List[int]) -> List[int]:
    """Bitwise XOR of two truth tables of equal length."""
    if len(a_tt) != len(b_tt):
        raise ValueError("Truth tables must have the same length for XOR.")
    return [int(a_tt[i]) ^ int(b_tt[i]) for i in range(len(a_tt))]


# ============================================================
# One-stage training (Steps 1–4 packaged)
# ============================================================

def train_stage_from_tt(
    target_tt: List[int],
    B: int,
    tau: float,
    K: int,
    T: int,
    R: int,
    step1: str = "exact",
    step3: str = "exact",
    use_espresso: bool = False,
) -> Tuple[Dict, List[int], List[float]]:
    """
    Trains one stage predictor for a target Boolean function given by truth table.

    Returns:
      (model, J, infl)
    where model predicts on the projected space {0,1}^{|J|}.
    """
    oracle = lambda x: target_tt[x]

    # Step 1: influences
    if step1 == "exact":
        infl = influences_exact_tt(target_tt, B)
    elif step1 == "oracle":
        infl = influences_oracle_uniform(oracle, B, T)
    else:
        raise ValueError("Unknown step1. Use 'exact' or 'oracle'.")

    # Step 2: select J_{tau,K}
    J = select_J_tau_K(infl, tau, K)

    # Step 3: build surrogate f_tau on J
    if step3 == "exact":
        f_tau_tt = f_tau_exact_marginal(target_tt, B, J)
    elif step3 == "mc":
        f_tau_tt = f_tau_monte_carlo(oracle, B, J, R)
    elif step3 == "fwht":
        f_tau_tt = f_tau_fwht(target_tt, B, J)
    else:
        raise ValueError("Unknown step3. Use 'exact' | 'mc' | 'fwht'.")

    # Step 4: optional Espresso
    model = learn_espresso(f_tau_tt) if use_espresso else {"expr": None, "g_tt": f_tau_tt}
    return model, J, infl


# ============================================================
# Accuracy
# ============================================================

def accuracy_full_from_preds(true_tt: List[int], pred_tt: List[int]) -> float:
    if len(true_tt) != len(pred_tt):
        raise ValueError("Truth tables must have the same length to compute accuracy.")
    correct = sum(1 for a, b in zip(true_tt, pred_tt) if a == b)
    return correct / len(true_tt)


# ============================================================
# Two-stage XOR residual boosting
# ============================================================

def run_m_stage_xor_boost(
    m=3,                 # number of stages
    seeds=10,
    B=12,
    S=6,
    tau=0.01,
    K=3,
    T=1000,
    R=200,
    step1="exact",
    step3="exact",
    use_espresso=False,
    verbose=False,
):
    """
    For each seed:
      - sample target f (here: random S-junta)
      - initialize H0(x)=0
      - for t=1..m:
          residual r_t = f XOR H_{t-1}
          train stage F_t ≈ r_t using Steps 1–4
          update H_t = H_{t-1} XOR F_t
      - report:
          acc(H_t, f) for each stage, and final acc(H_m,f)
          plus fraction of ones in f, residuals, and stage predictors (if verbose)

    Note: By XOR algebra,
      acc(H_t, f) == acc(F_t, r_t)  (pointwise equality of error events).
    """
    if isinstance(seeds, int):
        seeds = list(range(seeds))

    # Store per-stage accuracies across seeds
    acc_H_by_stage = [[] for _ in range(m)]          # acc(H_t, f)
    acc_stage_on_residual = [[] for _ in range(m)]   # acc(F_t, r_t)

    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed)

        # Target function f
        f_tt, true_junta_bits = random_s_junta_truth_table(B, S)

        # Running predictor H_0 = 0
        H_tt = [0] * (1 << B)

        if verbose:
            print(f"\n[seed={seed}]")
            print(f"  True junta bits (hidden): {true_junta_bits}")
            print(f"  frac_ones(f) = {ones_fraction(f_tt):.6f}")

        for t in range(1, m + 1):
            # Residual r_t = f XOR H_{t-1}
            r_tt = xor_tt(f_tt, H_tt)

            # Train stage t on residual
            modelT, JT, _ = train_stage_from_tt(
                r_tt, B=B, tau=tau, K=K, T=T, R=R,
                step1=step1, step3=step3, use_espresso=use_espresso
            )
            F_t_hat_tt = predict_stage_full_tt(B, modelT, JT)

            # Stage accuracy on residual
            acc_Ft_on_rt = accuracy_full_from_preds(r_tt, F_t_hat_tt)

            # Update running predictor
            H_tt = xor_tt(H_tt, F_t_hat_tt)

            # Overall accuracy after this stage
            acc_Ht = accuracy_full_from_preds(f_tt, H_tt)

            acc_stage_on_residual[t - 1].append(acc_Ft_on_rt)
            acc_H_by_stage[t - 1].append(acc_Ht)

            if verbose:
                print(f"  stage {t}: |J|={len(JT)} JT={JT}")
                print(f"    frac_ones(residual r_{t}) = {ones_fraction(r_tt):.6f}")
                print(f"    frac_ones(F_{t})          = {ones_fraction(F_t_hat_tt):.6f}")
                print(f"    acc(F_{t}, r_{t})         = {acc_Ft_on_rt:.6f}")
                print(f"    acc(H_{t}, f)             = {acc_Ht:.6f}")

    # ---- Print summary
    print("\n=== M-STAGE XOR-RESIDUAL BOOST RESULT ===")
    print(f"Stages (m): {m}")
    print(f"Seeds:      {len(seeds)}")
    for t in range(1, m + 1):
        print(f"Stage {t}: Avg acc(F_{t}, r_{t}) = {float(np.mean(acc_stage_on_residual[t-1])):.6f} "
              f"| Avg acc(H_{t}, f) = {float(np.mean(acc_H_by_stage[t-1])):.6f}")
    print(f"Final:   Avg acc(H_{m}, f) = {float(np.mean(acc_H_by_stage[m-1])):.6f}")

def run_two_stage_xor_boost(
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
    verbose=False,
):
    """
    For each seed:
      - sample target f (here: random S-junta)
      - train stage 1: F ≈ f using Steps 1–4
      - compute residual truth table g = f XOR F
      - train stage 2: G ≈ g using Steps 1–4
      - final predictor H = F XOR G
      - report accuracies (F, G-on-residual, H) averaged over seeds
    """
    if isinstance(seeds, int):
        seeds = list(range(seeds))

    acc_F_list = []
    acc_G_on_g_list = []
    acc_H_list = []

    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed)

        # Target function f
        f_tt, true_junta_bits = random_s_junta_truth_table(B, S)

        # ---- Stage 1: learn F for f
        modelF, JF, _ = train_stage_from_tt(
            f_tt, B=B, tau=tau, K=K, T=T, R=R, step1=step1, step3=step3, use_espresso=use_espresso
        )
        F_hat_tt = predict_stage_full_tt(B, modelF, JF)
        acc_F = accuracy_full_from_preds(f_tt, F_hat_tt)

        # ---- Residual: g = f XOR F
        g_tt = xor_tt(f_tt, F_hat_tt)

        # ---- Stage 2: learn G for g
        modelG, JG, _ = train_stage_from_tt(
            g_tt, B=B, tau=tau, K=K, T=T, R=R, step1=step1, step3=step3, use_espresso=use_espresso
        )
        G_hat_tt = predict_stage_full_tt(B, modelG, JG)
        acc_G_on_g = accuracy_full_from_preds(g_tt, G_hat_tt)

        # ---- Final: H = F XOR G
        H_hat_tt = xor_tt(F_hat_tt, G_hat_tt)
        acc_H = accuracy_full_from_preds(f_tt, H_hat_tt)

        acc_F_list.append(acc_F)
        acc_G_on_g_list.append(acc_G_on_g)
        acc_H_list.append(acc_H)

        # ---- Fraction of label-1s
        frac_f = ones_fraction(f_tt)
        frac_F = ones_fraction(F_hat_tt)
        frac_g = ones_fraction(g_tt)
        frac_G = ones_fraction(G_hat_tt)

        if verbose:
            print(f"  frac_ones(f) = {frac_f:.6f}")
            print(f"  frac_ones(F) = {frac_F:.6f}")
            print(f"  frac_ones(g) = {frac_g:.6f}")
            print(f"  frac_ones(G) = {frac_G:.6f}")

        if verbose:
            print(f"\n[seed={seed}]")
            print(f"  True junta bits (hidden): {true_junta_bits}")
            print(f"  Stage1 |J|={len(JF)} JF={JF}  acc(F,f)={acc_F:.4f}")
            print(f"  Stage2 |J|={len(JG)} JG={JG}  acc(G,g)={acc_G_on_g:.4f}")
            print(f"  Final  acc(H,f)={acc_H:.4f}")

    print("\n=== TWO-STAGE XOR-RESIDUAL BOOST RESULT ===")
    print(f"Seeds: {len(seeds)}")
    print(f"Avg acc(F, f):       {float(np.mean(acc_F_list)):.6f}")
    print(f"Avg acc(G, residual):{float(np.mean(acc_G_on_g_list)):.6f}")
    print(f"Avg acc(H, f):       {float(np.mean(acc_H_list)):.6f}")


# ============================================================
# Run experiment
# ============================================================

# if __name__ == "__main__":
#     run_two_stage_xor_boost(
#         seeds=10,          # number of random seeds (or a list of ints)
#         B=12,
#         S=6,
#         tau=0.01,
#         K=3,
#         T=1000,
#         R=200,
#         step1="exact",     # exact | oracle
#         step3="exact",      # exact | mc | fwht
#         use_espresso=False,
#         verbose=False,
#     )

if __name__ == "__main__":
    run_m_stage_xor_boost(
        m=50,
        seeds=10,
        B=15,
        S=12,
        tau=0.01,
        K=6,
        T=1000,
        R=200,
        step1="exact",     # exact | oracle
        step3="exact",     # exact | mc | fwht
        use_espresso=False,
        verbose=False,
    )