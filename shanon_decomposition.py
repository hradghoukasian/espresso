import time
import random
from typing import List

from pyeda.inter import exprvars
from espresso_minimization import minimize_truth_table_espresso


def random_tt(n_vars: int, seed: int = 0) -> str:
    rng = random.Random(seed)
    return "".join(rng.choice("01") for _ in range(1 << n_vars))


def build_cofactor_tts(f_tt: str, n_total: int, k_split: int) -> List[str]:
    """
    IMPORTANT: This slicing assumes PyEDA truth-table ordering:
      idx = x[0] + 2*x[1] + ... + 2^{n-1}*x[n-1]
    so x[0] is the fast/LSB bit and x[n-1] is the slow/MSB bit.

    If we split with m = n_total - k_split, then:
      data vars = x[0]..x[m-1]  (low bits)
      sel vars  = x[m]..x[n-1]  (high bits)

    Under this ordering, fixing sel corresponds to contiguous blocks in f_tt.
    """
    m = n_total - k_split
    num_blocks = 1 << k_split
    block_size = 1 << m

    cofactors = []
    for sel in range(num_blocks):
        start = sel * block_size
        end = start + block_size
        cofactors.append(f_tt[start:end])  # truth table over x[0]..x[m-1]
    return cofactors


def eval_expr_tt(expr, X, m: int) -> List[int]:
    """
    Evaluate expr on all 2^m assignments using PyEDA's ordering:
    idx = x[0] + 2*x[1] + ... + 2^{m-1}*x[m-1]
    """
    out = [0] * (1 << m)
    for idx in range(1 << m):
        point = {X[k]: (idx >> k) & 1 for k in range(m)}  # x[0] is LSB
        val = expr.restrict(point)
        out[idx] = 1 if val.is_one() else 0
    return out


def hierarchical_espresso(f_tt: str, n_total: int = 20, k_split: int = 10):
    assert len(f_tt) == (1 << n_total)
    assert set(f_tt).issubset({"0", "1"}), "Use only 0/1 for guaranteed 100% accuracy."

    m = n_total - k_split
    num_blocks = 1 << k_split
    block_size = 1 << m

    print(f"\n=== Hierarchical Espresso (split {k_split} + {m}) ===")
    print(f"num cofactors = 2^{k_split} = {num_blocks}")
    print(f"each cofactor has 2^{m} = {block_size} rows\n")

    # Build cofactors (exact, by slicing contiguous blocks)
    t0 = time.perf_counter()
    cofactor_tts = build_cofactor_tts(f_tt, n_total, k_split)
    t_build = time.perf_counter() - t0
    print(f"[time] build cofactors: {t_build:.3f}s")

    # Espresso each cofactor + precompute its predicted truth table on 2^m inputs
    X_m = exprvars("x", m)
    pred_tables = [None] * num_blocks
    total_espresso_time = 0.0

    t0 = time.perf_counter()
    for sel in range(num_blocks):
        expr, stats = minimize_truth_table_espresso(cofactor_tts[sel], verbose=False)
        total_espresso_time += stats["elapsed"]

        # Precompute outputs on all 2^m data inputs (fast lookup later)
        pred_tables[sel] = eval_expr_tt(expr, X_m, m)

    t_espresso_wall = time.perf_counter() - t0

    print(f"[time] espresso sum elapsed: {total_espresso_time:.3e}s")
    print(f"[time] espresso wall-clock:  {t_espresso_wall:.3f}s")

    # Exact recombination + accuracy over all 2^n_total
    # Under PyEDA ordering:
    #   data = low m bits  -> x[0]..x[m-1]
    #   sel  = high k bits -> x[m]..x[n-1]
    t0 = time.perf_counter()
    correct = 0
    total = 1 << n_total

    for idx in range(total):
        data = idx & (block_size - 1)   # low bits
        sel = idx >> m                 # high bits

        y_hat = pred_tables[sel][data]
        y = 1 if f_tt[idx] == "1" else 0
        correct += (y_hat == y)

    acc = correct / total
    t_acc = time.perf_counter() - t0

    print(f"[time] accuracy eval over 2^{n_total}: {t_acc:.3f}s")
    print(f"\naccuracy: {acc:.6f}")

    if abs(acc - 1.0) < 1e-12:
        print("✅ 100% accuracy (exact decomposition)\n")
    else:
        print("⚠️ Not 100%. If this happens now, it usually means the TT ordering used to build f_tt is different.\n")

    return acc


def main():
    n_total = 15
    k_split = 10
    seed = 0

    f_tt = random_tt(n_total, seed=seed)
    hierarchical_espresso(f_tt, n_total=n_total, k_split=k_split)


if __name__ == "__main__":
    main()
