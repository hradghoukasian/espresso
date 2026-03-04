"""
friedgut_junta_pipeline_partial_tt.py

multi-stage XOR residual learning with Espresso-based stage learners, WITHOUT access to the full 2^B truth table.

We have:
  D_train = {(x_j, f(x_j))}_{j=1..T_train}
  D_test  = {(x_j, f(x_j))}_{j=1..T_test}

Algorithm (m stages):
  H_0(x) = 0
  r_t(x) = f(x) XOR H_{t-1}(x)

  1) Estimate influences Inf_i using ONLY observed neighbor pairs in the training set:

  2) Select J_t = top-K bits among {i : Inf_i > tau}, ties broken randomly.
     If fewer than K, keep smaller J_t.

  3) Build projected surrogate g_t over {0,1}^{|J_t|} via empirical majority from training residuals:
        for each u:
            if u seen -> majority label (tie broken randomly)
            if u unseen -> DON'T CARE '-'

  4) Learn Espresso-minimized stage function from that partial truth table.

Stage representation:
  - bits: J_t (list of bit indices)
  - tt_full: length 2^|J_t| list of 0/1 predictions for every projection pattern
             (we fill DON'T CARE patterns using random 0/1 so we have a total function)
  - expr_str: string for the minimized expression

Prediction:
  H_m(x) = XOR_{t=1..m} F_t(x)

Requires:
  pip install pyeda
"""

from __future__ import annotations
import time
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple


# ============================================================
# PyEDA import
# ============================================================

PYEDA_OK = True
ESPRESSO_EXPR_AVAILABLE = True
PYEDA_IMPORT_ERR = None

try:
    from pyeda.inter import exprvars, Or, And
    try:
        from pyeda.inter import espresso_exprs  # expression-level wrapper
    except Exception as e:
        espresso_exprs = None
        ESPRESSO_EXPR_AVAILABLE = False
except Exception as e:
    PYEDA_OK = False
    ESPRESSO_EXPR_AVAILABLE = False
    PYEDA_IMPORT_ERR = e
    exprvars = None
    Or = None
    And = None
    espresso_exprs = None


# ============================================================
# Bit utilities
# ============================================================

def proj_index(x: int, cols: Sequence[int]) -> int:
    """Project x onto bits in cols (in given order) and return as an int in [0, 2^|cols|)."""
    idx = 0
    for j, bit in enumerate(cols):
        if (x >> bit) & 1:
            idx |= 1 << j
    return idx


# ============================================================
# Synthetic target function: S-junta
# ============================================================

@dataclass(frozen=True)
class SJunta:
    B: int
    junta_bits: List[int]      # length S
    junta_tt: List[int]        # length 2^S

    def __call__(self, x: int) -> int:
        u = proj_index(x, self.junta_bits)
        return self.junta_tt[u]


def make_random_s_junta(B: int, S: int, rng: random.Random) -> SJunta:
    junta_bits = sorted(rng.sample(range(B), S))
    junta_tt = [rng.randint(0, 1) for _ in range(1 << S)]
    return SJunta(B=B, junta_bits=junta_bits, junta_tt=junta_tt)


def sample_dataset(f: SJunta, T: int, rng: random.Random) -> Tuple[List[int], List[int]]:
    """Uniform random samples over {0,1}^B, with labels from f."""
    # xs = [rng.randrange(1 << f.B) for _ in range(T)]    # sampling with replacement
    xs = rng.sample(range(1 << f.B), T)   # sampling w/o replacement
    ys = [f(x) for x in xs]
    return xs, ys


# ============================================================
# Espresso stage representation
# ============================================================

@dataclass
class Stage:
    bits: List[int]          # J_t
    tt_full: List[int]       # length 2^k (k=len(bits)), definitive 0/1
    expr_str: str            # minimized expression string (or fallback)

    def predict_one(self, x: int) -> int:
        k = len(self.bits)
        if k == 0:
            return 0
        u = proj_index(x, self.bits)
        return self.tt_full[u]


# ============================================================
# Model = XOR of stages
# ============================================================

def predict_model(stages: List[Stage], x: int) -> int:
    y = 0
    for st in stages:
        y ^= st.predict_one(x)
    return y


def accuracy(stages: List[Stage], xs: Sequence[int], ys: Sequence[int]) -> float:
    correct = 0
    for x, y in zip(xs, ys):
        if predict_model(stages, x) == y:
            correct += 1
    return correct / max(1, len(xs))


# ============================================================
# Influence estimation from observed neighbor pairs
# ============================================================

def influences_from_dataset_pairs(
    B: int,
    x_train: Sequence[int],
    residual_map: Dict[int, int],
) -> Tuple[List[float], List[int]]:
    """
    For each bit i:
      P_i = {x in train : x^e_i also in train}
      Inf_i = average of 1{r(x)!=r(x^e_i)} over P_i (0 if |P_i|=0)

    Returns:
      influences: length B
      pair_counts: length B (|P_i|)
    """
    xset = set(x_train)
    influences = [0.0] * B
    pair_counts = [0] * B

    for i in range(B):
        mask = 1 << i
        mismatches = 0
        cnt = 0
        for x in x_train:
            xn = x ^ mask
            if xn in xset:
                cnt += 1
                if residual_map[x] != residual_map[xn]:
                    mismatches += 1
        pair_counts[i] = cnt
        influences[i] = (mismatches / cnt) if cnt > 0 else 0.0

    return influences, pair_counts


# ============================================================
# Select J_t = top-K among {i : Inf_i > tau}, ties broken randomly
# If fewer than K, keep smaller J_t.
# ============================================================

def select_topK_with_threshold(
    influences: Sequence[float],
    K: int,
    tau: float,
    rng: random.Random,
) -> List[int]:
    candidates = [i for i, inf in enumerate(influences) if inf > tau]
    rng.shuffle(candidates)  # random tie-breaking
    candidates.sort(key=lambda i: influences[i], reverse=True)
    return candidates[: min(K, len(candidates))]


# ============================================================
# Build projected surrogate g_t with DON'T CARE for unseen patterns
# Tie-breaking in majorities: random
# ============================================================

def build_projected_majority_tt_with_dc(
    x_train: Sequence[int],
    r_train: Sequence[int],
    J: Sequence[int],
    rng: random.Random,
) -> Tuple[List[str], int]:
    """
    Returns:
      tt_list: length 2^k list over {'0','1','-'} where '-' means don't care.
      k: |J|
    """
    k = len(J)
    if k == 0:
        c1 = sum(r_train)
        c0 = len(r_train) - c1
        if c1 > c0:
            return ["1"], 0
        if c0 > c1:
            return ["0"], 0
        return [str(rng.randint(0, 1))], 0

    size = 1 << k
    count0 = [0] * size
    count1 = [0] * size

    for x, r in zip(x_train, r_train):
        u = proj_index(x, J)
        if r == 1:
            count1[u] += 1
        else:
            count0[u] += 1

    tt_list: List[str] = []
    for u in range(size):
        c0, c1 = count0[u], count1[u]
        if c0 == 0 and c1 == 0:
            tt_list.append("-")  # unseen -> don't care
        elif c1 > c0:
            tt_list.append("1")
        elif c0 > c1:
            tt_list.append("0")
        else:
            tt_list.append(str(rng.randint(0, 1)))  # tie -> random

    return tt_list, k


def format_selected_influences(J: List[int], infl: Sequence[float], max_show: int = 20) -> str:
    pairs = [(i, infl[i]) for i in J]
    pairs.sort(key=lambda p: p[1], reverse=True)
    pairs = pairs[:max_show]
    return ", ".join([f"{i}:{v:.4f}" for i, v in pairs])


# ============================================================
# Espresso minimization from a 0/1/- table using expression-level wrapper.
# Fallback if espresso_exprs is missing or fails.
# ============================================================

def learn_espresso_stage_from_tt(
    tt_list: List[str],
    k: int,
    rng: random.Random,
) -> Tuple[List[int], str]:
    """
    Input:
      tt_list: length 2^k list over {'0','1','-'}
      k: number of variables
    Output:
      tt_full: length 2^k list of 0/1 for fast prediction
      expr_str: minimized expression string (or fallback label)

    NOTE: We must return a total function (0/1 everywhere) to predict on arbitrary x.
          So for '-' entries we fill them randomly.
    """
    if k == 0:
        val = 1 if tt_list[0] == "1" else 0
        return [val], tt_list[0]

    size = 1 << k

    # Fill don't-cares so the stage is a total function for prediction
    tt_filled = [ch if ch != "-" else str(rng.randint(0, 1)) for ch in tt_list]
    tt_full_from_fill = [1 if ch == "1" else 0 for ch in tt_filled]

    # If PyEDA isn't available, return filled table (no minimization)
    if (not PYEDA_OK) or (not ESPRESSO_EXPR_AVAILABLE) or (espresso_exprs is None):
        reason = "PyEDA/espresso_exprs unavailable; using filled table (no minimization)"
        return tt_full_from_fill, reason

    # Build ON-set SOP for positions where output is 1.
    X = exprvars("x", k)
    on_terms = []
    for u, ch in enumerate(tt_filled):
        if ch != "1":
            continue
        lits = [(X[j] if ((u >> j) & 1) else ~X[j]) for j in range(k)]
        on_terms.append(And(*lits))

    if not on_terms:
        return [0] * size, "0"

    expr = Or(*on_terms)

    # Try Espresso minimization; if it fails, fall back
    try:
        minimized = espresso_exprs(expr)[0]
    except Exception as e:
        reason = f"espresso_exprs failed ({type(e).__name__}: {e}); using filled table (no minimization)"
        return tt_full_from_fill, reason

    # Precompute minimized table
    tt_full: List[int] = []
    for u in range(size):
        assignment = {X[j]: ((u >> j) & 1) for j in range(k)}
        v = minimized.restrict(assignment)
        tt_full.append(1 if v.is_one() else 0)

    return tt_full, str(minimized)


# ============================================================
# Training loop (NOW returns stage_test_acc dict for tracked stages)
# ============================================================

def train_multistage_xor_espresso(
    B: int,
    K: int,
    m: int,
    tau: float,
    x_train: List[int],
    y_train: List[int],
    x_test: Optional[List[int]] = None,
    y_test: Optional[List[int]] = None,
    seed: int = 0,
    verbose: bool = True,
    track_stages: Optional[List[int]] = None,
) -> Tuple[List[Stage], Dict[int, float]]:
    rng = random.Random(seed)
    stages: List[Stage] = []

    if track_stages is None:
        track_stages = []
    track_set = set(track_stages)
    stage_test_acc: Dict[int, float] = {}

    if verbose and (not PYEDA_OK):
        print(f"[warn] PyEDA import failed: {PYEDA_IMPORT_ERR}")
        print("[warn] Will run WITHOUT Espresso minimization (still trains/predicts).")

    if verbose and PYEDA_OK and (not ESPRESSO_EXPR_AVAILABLE):
        print("[warn] pyeda.inter.espresso_exprs not available in your PyEDA build.")
        print("[warn] Will run WITHOUT Espresso minimization (still trains/predicts).")

    for t in range(1, m + 1):
        # 1) residual on training samples
        r_train: List[int] = []
        residual_map: Dict[int, int] = {}
        for x, y in zip(x_train, y_train):
            ht = predict_model(stages, x)
            r = y ^ ht
            r_train.append(r)
            residual_map[x] = r

        # 2) influences from observed neighbor pairs only
        infl, pair_counts = influences_from_dataset_pairs(B, x_train, residual_map)

        # 3) select J_t
        J = select_topK_with_threshold(infl, K=K, tau=tau, rng=rng)

        # 4) projected majority over J (unseen -> don't care)
        tt_list, k = build_projected_majority_tt_with_dc(x_train, r_train, J, rng=rng)

        # 5) learn stage
        tt_full, expr_str = learn_espresso_stage_from_tt(tt_list, k=k, rng=rng)
        stages.append(Stage(bits=list(J), tt_full=tt_full, expr_str=expr_str))

        # Track test accuracy at specific stages
        if (x_test is not None) and (y_test is not None) and (t in track_set):
            stage_test_acc[t] = accuracy(stages, x_test, y_test)

        # Logging
        if verbose:
            train_acc = accuracy(stages, x_train, y_train)
            msg = f"[stage {t:02d}] |J|={len(J)} tau={tau:.4g} train_acc={train_acc:.4f}"

            nonzero_pairs = sum(1 for c in pair_counts if c > 0)
            avg_pairs = sum(pair_counts) / max(1, B)
            msg += f"  pairbits_nonzero={nonzero_pairs}/{B} avg_|P_i|={avg_pairs:.2f}"

            if len(J) > 0:
                size = 1 << len(J)
                seen = sum(1 for ch in tt_list if ch != "-")
                msg += f"  proj_seen={seen}/{size}"
                msg += f"  sel_infl=[{format_selected_influences(J, infl)}]"
            else:
                msg += "  sel_infl=[]"

            if x_test is not None and y_test is not None:
                test_acc_now = accuracy(stages, x_test, y_test)
                msg += f"  test_acc={test_acc_now:.4f}"

            if ("no minimization" in expr_str) or ("failed" in expr_str):
                msg += "  [espresso:fallback]"

            print(msg)

    return stages, stage_test_acc


# ============================================================
# Multi-seed evaluation
# ============================================================

def mean_std(vals: List[float]) -> Tuple[float, float]:
    if len(vals) == 0:
        return 0.0, 0.0
    mu = sum(vals) / len(vals)
    var = sum((v - mu) ** 2 for v in vals) / len(vals)  # population variance
    return mu, math.sqrt(var)


def run_many_seeds(
    num_seeds: int,
    base_seed: int,
    B: int,
    S: int,
    T_train: int,
    T_test: int,
    K: int,
    m: int,
    tau: float,
    stage_points: List[int],
    verbose_each_seed: bool = False,
) -> None:

    for sp in stage_points:
        if sp < 1 or sp > m:
            raise ValueError(f"stage_points contains {sp}, but m={m}. Use sp<=m.")

    train_accs: List[float] = []
    test_accs: List[float] = []
    stage_test_accs: Dict[int, List[float]] = {sp: [] for sp in stage_points}
    train_times: List[float] = []

    for s in range(num_seeds):
        seed = base_seed + s
        rng = random.Random(seed)

        f = make_random_s_junta(B=B, S=S, rng=rng)
        x_train, y_train = sample_dataset(f, T_train, rng=rng)
        x_test, y_test = sample_dataset(f, T_test, rng=rng)

        if verbose_each_seed:
            print("\n" + "=" * 80)
            print(f"[seed {seed}] True junta bits: {f.junta_bits}")

        # START TIMING
        t0 = time.perf_counter()

        stages, stage_map = train_multistage_xor_espresso(
            B=B,
            K=K,
            m=m,
            tau=tau,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            seed=seed,
            verbose=verbose_each_seed,
            track_stages=stage_points,
        )

        # END TIMING
        t1 = time.perf_counter()
        elapsed = t1 - t0
        train_times.append(elapsed)

        final_train = accuracy(stages, x_train, y_train)
        final_test = accuracy(stages, x_test, y_test)

        train_accs.append(final_train)
        test_accs.append(final_test)

        for sp in stage_points:
            stage_test_accs[sp].append(stage_map.get(sp, float("nan")))

        tracked_str = "  ".join(
            [f"test@{sp}={stage_map.get(sp, float('nan')):.4f}" for sp in stage_points]
        )

        print(
            f"[seed {seed}] "
            f"train={final_train:.4f} "
            f"test={final_test:.4f} "
            f"time={elapsed:.2f}s  "
            f"{tracked_str}"
        )

    # =======================
    # Summary statistics
    # =======================

    mu_tr, sd_tr = mean_std(train_accs)
    mu_te, sd_te = mean_std(test_accs)
    mu_time, sd_time = mean_std(train_times)

    print("\n" + "#" * 80)
    print(f"Summary over {num_seeds} seeds (base_seed={base_seed})")
    print(f"B={B}, S={S}, T_train={T_train}, T_test={T_test}, K={K}, m={m}, tau={tau}")
    print()
    print(f"Final TRAIN accuracy: mean={mu_tr:.4f}, std={sd_tr:.4f}")
    print(f"Final TEST  accuracy: mean={mu_te:.4f}, std={sd_te:.4f}")
    print()
    print(f"Training time (full multi-stage circuit):")
    print(f"  mean={mu_time:.2f}s, std={sd_time:.2f}s")
    print(f"  avg per stage ≈ {mu_time / m:.3f}s")
    print()

    for sp in stage_points:
        vals = [v for v in stage_test_accs[sp] if not math.isnan(v)]
        mu_sp, sd_sp = mean_std(vals)
        print(f"TEST accuracy at stage {sp:>3}: mean={mu_sp:.4f}, std={sd_sp:.4f}")

    print("#" * 80 + "\n")

# ============================================================
# Main
# ============================================================

def main():
    # Initial dimension and true junta size
    B = 15
    S = 12

    # Samples
    T_train = 2**B
    T_test = 2**B

    # Algorithm hyperparameters
    K = 6
    m = 20
    tau = 0.02

    # Multi-seed experiment settings
    num_seeds = 10
    base_seed = 42

    # stage 1, 5, 20
    stage_points = [1, 5, 20]

    run_many_seeds(
        num_seeds=num_seeds,
        base_seed=base_seed,
        B=B,
        S=S,
        T_train=T_train,
        T_test=T_test,
        K=K,
        m=m,
        tau=tau,
        stage_points=stage_points,
        verbose_each_seed=False,  # True prints per-stage logs for each seed
    )

if __name__ == "__main__":
    main()