"""
friedgut_junta_pipeline_oracle_ablation.py

Multi-stage XOR residual learning with Espresso-based stage learners, with
independent ablations for:

  1) Influence computation:
       - "sample": use only observed neighbor pairs in D_train
       - "oracle": use exact/full-truth-table influence of the residual

  2) Marginalization / projected surrogate construction:
       - "sample": majority over observed training residuals only
       - "oracle": exact/full-truth-table majority over all completions

This extends the partial-truth-table Algorithm 3 setup from the attached Espresso
PDF, where the baseline uses:
  - sample-based influence from matched pairs P_i
  - sample-based majority on observed A_t(u) only

Requires:
  pip install pyeda

Typical ablation settings:
  - sample/sample   : original Algorithm 3 baseline
  - oracle/sample   : exact influence only
  - sample/oracle   : exact marginalization only
  - oracle/oracle   : both exact

Notes:
  - Evaluation is always done on D_test (or any provided test set).
  - "oracle" is intended for synthetic settings where we truly can evaluate f(x)
    on the whole cube (e.g., random S-juntas).
"""

from __future__ import annotations
import time
import math
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple


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
    """Project x onto bits in cols (in given order) and return int in [0, 2^|cols|)."""
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


def sample_dataset(f: Callable[[int], int], B: int, T: int, rng: random.Random) -> Tuple[List[int], List[int]]:
    """Uniform random samples over {0,1}^B, sampled without replacement."""
    if T > (1 << B):
        raise ValueError(f"T={T} exceeds full cube size 2^B={1 << B}.")
    xs = rng.sample(range(1 << B), T)
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
# Influence estimation from observed neighbor pairs (sample mode)
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
# Exact influence over the full truth table (oracle mode)
# ============================================================

def exact_influences_full_cube(
    B: int,
    residual_full: Sequence[int],
) -> Tuple[List[float], List[int]]:
    """
    Exact influence:
      Inf_i(r) = Pr_x[r(x) != r(x^e_i)]

    Computed over the entire cube.

    Returns:
      influences: length B
      pair_counts: length B, set to full-cube count 2^B for logging compatibility
    """
    size = 1 << B
    influences = [0.0] * B
    pair_counts = [size] * B

    for i in range(B):
        mask = 1 << i
        mismatches = 0
        for x in range(size):
            if residual_full[x] != residual_full[x ^ mask]:
                mismatches += 1
        influences[i] = mismatches / size

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
# (sample mode)
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


# ============================================================
# Exact projected majority over the full truth table (oracle mode)
# ============================================================

def build_projected_majority_tt_oracle(
    B: int,
    residual_full: Sequence[int],
    J: Sequence[int],
    rng: random.Random,
) -> Tuple[List[str], int]:
    """
    Exact projected surrogate:
      g(u) = Maj{ r(x) : proj_J(x) = u }

    Since we use the full cube, every u is seen; so no '-' appears.
    We still return a List[str] over {'0','1'} for compatibility.
    """
    k = len(J)
    if k == 0:
        c1 = sum(residual_full)
        c0 = len(residual_full) - c1
        if c1 > c0:
            return ["1"], 0
        if c0 > c1:
            return ["0"], 0
        return [str(rng.randint(0, 1))], 0

    size = 1 << k
    count0 = [0] * size
    count1 = [0] * size

    full_size = 1 << B
    for x in range(full_size):
        u = proj_index(x, J)
        r = residual_full[x]
        if r == 1:
            count1[u] += 1
        else:
            count0[u] += 1

    tt_list: List[str] = []
    for u in range(size):
        c0, c1 = count0[u], count1[u]
        if c1 > c0:
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
# Residual helpers
# ============================================================

def compute_residuals_on_train(
    stages: List[Stage],
    x_train: Sequence[int],
    y_train: Sequence[int],
) -> Tuple[List[int], Dict[int, int]]:
    r_train: List[int] = []
    residual_map: Dict[int, int] = {}
    for x, y in zip(x_train, y_train):
        ht = predict_model(stages, x)
        r = y ^ ht
        r_train.append(r)
        residual_map[x] = r
    return r_train, residual_map


def compute_residuals_on_full_cube(
    B: int,
    stages: List[Stage],
    oracle_f: Callable[[int], int],
) -> List[int]:
    size = 1 << B
    residual_full = [0] * size
    for x in range(size):
        residual_full[x] = oracle_f(x) ^ predict_model(stages, x)
    return residual_full


# ============================================================
# Unified oracle/sample wrappers
# ============================================================

def compute_influences(
    B: int,
    x_train: Sequence[int],
    residual_map: Dict[int, int],
    influence_mode: str,
    residual_full: Optional[Sequence[int]] = None,
) -> Tuple[List[float], List[int]]:
    if influence_mode == "sample":
        return influences_from_dataset_pairs(B, x_train, residual_map)

    if influence_mode == "oracle":
        if residual_full is None:
            raise ValueError("residual_full must be provided when influence_mode='oracle'.")
        return exact_influences_full_cube(B, residual_full)

    raise ValueError(f"Unknown influence_mode={influence_mode!r}. Use 'sample' or 'oracle'.")


def build_projected_surrogate_tt(
    B: int,
    x_train: Sequence[int],
    r_train: Sequence[int],
    J: Sequence[int],
    rng: random.Random,
    marginal_mode: str,
    residual_full: Optional[Sequence[int]] = None,
) -> Tuple[List[str], int]:
    if marginal_mode == "sample":
        return build_projected_majority_tt_with_dc(x_train, r_train, J, rng=rng)

    if marginal_mode == "oracle":
        if residual_full is None:
            raise ValueError("residual_full must be provided when marginal_mode='oracle'.")
        return build_projected_majority_tt_oracle(B, residual_full, J, rng=rng)

    raise ValueError(f"Unknown marginal_mode={marginal_mode!r}. Use 'sample' or 'oracle'.")


# ============================================================
# Training loop with oracle ablations
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
    influence_mode: str = "sample",   # "sample" or "oracle"
    marginal_mode: str = "sample",    # "sample" or "oracle"
    oracle_f: Optional[Callable[[int], int]] = None,
) -> Tuple[List[Stage], Dict[int, float]]:
    rng = random.Random(seed)
    stages: List[Stage] = []

    if track_stages is None:
        track_stages = []
    track_set = set(track_stages)
    stage_test_acc: Dict[int, float] = {}

    if influence_mode not in {"sample", "oracle"}:
        raise ValueError("influence_mode must be 'sample' or 'oracle'.")
    if marginal_mode not in {"sample", "oracle"}:
        raise ValueError("marginal_mode must be 'sample' or 'oracle'.")

    need_oracle = (influence_mode == "oracle") or (marginal_mode == "oracle")
    if need_oracle and (oracle_f is None):
        raise ValueError("oracle_f must be provided when using any oracle mode.")

    if verbose and (not PYEDA_OK):
        print(f"[warn] PyEDA import failed: {PYEDA_IMPORT_ERR}")
        print("[warn] Will run WITHOUT Espresso minimization (still trains/predicts).")

    if verbose and PYEDA_OK and (not ESPRESSO_EXPR_AVAILABLE):
        print("[warn] pyeda.inter.espresso_exprs not available in your PyEDA build.")
        print("[warn] Will run WITHOUT Espresso minimization (still trains/predicts).")

    if verbose:
        print(f"[config] influence_mode={influence_mode} marginal_mode={marginal_mode}")

    for t in range(1, m + 1):
        # 1) residual on training samples
        r_train, residual_map = compute_residuals_on_train(stages, x_train, y_train)

        # Optional full-cube residual for oracle computations
        residual_full: Optional[List[int]] = None
        if need_oracle:
            residual_full = compute_residuals_on_full_cube(B, stages, oracle_f)

        # 2) influences
        infl, pair_counts = compute_influences(
            B=B,
            x_train=x_train,
            residual_map=residual_map,
            influence_mode=influence_mode,
            residual_full=residual_full,
        )

        # 3) select J_t
        J = select_topK_with_threshold(infl, K=K, tau=tau, rng=rng)

        # 4) projected surrogate over J
        tt_list, k = build_projected_surrogate_tt(
            B=B,
            x_train=x_train,
            r_train=r_train,
            J=J,
            rng=rng,
            marginal_mode=marginal_mode,
            residual_full=residual_full,
        )

        # 5) learn stage with Espresso
        tt_full, expr_str = learn_espresso_stage_from_tt(tt_list, k=k, rng=rng)
        stages.append(Stage(bits=list(J), tt_full=tt_full, expr_str=expr_str))

        # Track stage-specific test accuracy
        if (x_test is not None) and (y_test is not None) and (t in track_set):
            stage_test_acc[t] = accuracy(stages, x_test, y_test)

        # Logging
        if verbose:
            train_acc = accuracy(stages, x_train, y_train)
            msg = f"[stage {t:02d}] |J|={len(J)} tau={tau:.4g} train_acc={train_acc:.4f}"

            if influence_mode == "sample":
                nonzero_pairs = sum(1 for c in pair_counts if c > 0)
                avg_pairs = sum(pair_counts) / max(1, B)
                msg += f"  pairbits_nonzero={nonzero_pairs}/{B} avg_|P_i|={avg_pairs:.2f}"
            else:
                msg += "  exact_influence=full_cube"

            if len(J) > 0:
                size = 1 << len(J)
                seen = sum(1 for ch in tt_list if ch != "-")
                msg += f"  proj_seen={seen}/{size}"
                msg += f"  sel_infl=[{format_selected_influences(J, infl)}]"
            else:
                msg += "  sel_infl=[]"

            if marginal_mode == "oracle":
                msg += "  proj_majority=full_cube"

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
    influence_mode: str = "sample",
    marginal_mode: str = "sample",
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
        x_train, y_train = sample_dataset(f, B=B, T=T_train, rng=rng)
        x_test, y_test = sample_dataset(f, B=B, T=T_test, rng=rng)

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
            influence_mode=influence_mode,
            marginal_mode=marginal_mode,
            oracle_f=f,
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
            f"mode=({influence_mode},{marginal_mode})  "
            f"train={final_train:.4f} "
            f"test={final_test:.4f} "
            f"time={elapsed:.2f}s  "
            f"{tracked_str}"
        )

    # Summary statistics
    mu_tr, sd_tr = mean_std(train_accs)
    mu_te, sd_te = mean_std(test_accs)
    mu_time, sd_time = mean_std(train_times)

    print("\n" + "#" * 80)
    print(f"Summary over {num_seeds} seeds (base_seed={base_seed})")
    print(
        f"B={B}, S={S}, T_train={T_train}, T_test={T_test}, "
        f"K={K}, m={m}, tau={tau}, "
        f"influence_mode={influence_mode}, marginal_mode={marginal_mode}"
    )
    print()
    print(f"Final TRAIN accuracy: mean={mu_tr:.4f}, std={sd_tr:.4f}")
    print(f"Final TEST  accuracy: mean={mu_te:.4f}, std={sd_te:.4f}")
    for sp in stage_points:
        mu_sp, sd_sp = mean_std(stage_test_accs[sp])
        print(f"TEST accuracy at stage {sp}: mean={mu_sp:.4f}, std={sd_sp:.4f}")
    print(f"Training time: mean={mu_time:.2f}s, std={sd_time:.2f}s")
    print("#" * 80)


# ============================================================
# Convenience helper: run the 4 ablation configurations
# ============================================================

def run_ablation_suite(
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
    configs = [
        ("sample", "sample"),
        ("oracle", "sample"),
        ("sample", "oracle"),
        ("oracle", "oracle"),
    ]

    for influence_mode, marginal_mode in configs:
        print("\n" + "=" * 100)
        print(f"Ablation config: influence_mode={influence_mode}, marginal_mode={marginal_mode}")
        print("=" * 100)
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
            influence_mode=influence_mode,
            marginal_mode=marginal_mode,
            verbose_each_seed=verbose_each_seed,
        )


# ============================================================
# Example main
# ============================================================

if __name__ == "__main__":
    # Example settings similar to your current script style
    B = 13
    S = 8
    T_train = 4000
    T_test = 4000
    K = 6
    m = 20
    tau = 0.02
    stage_points = [1, 5, 20]

    num_seeds = 10
    base_seed = 42

    # # --------------------------------------------------------
    # # Run one specific configuration
    # # --------------------------------------------------------
    # print("\nRunning a single configuration...\n")
    # run_many_seeds(
    #     num_seeds=num_seeds,
    #     base_seed=base_seed,
    #     B=B,
    #     S=S,
    #     T_train=T_train,
    #     T_test=T_test,
    #     K=K,
    #     m=m,
    #     tau=tau,
    #     stage_points=stage_points,
    #     influence_mode="sample",   # change to "oracle" if desired
    #     marginal_mode="sample",    # change to "oracle" if desired
    #     verbose_each_seed=False,
    # )

    # --------------------------------------------------------
    # Run all 4 ablations
    # --------------------------------------------------------
    print("\nRunning the full 4-way ablation suite...\n")
    run_ablation_suite(
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
        verbose_each_seed=False,
    )