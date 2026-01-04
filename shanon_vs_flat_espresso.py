# shanon_vs_normal.py
# hierarchical_vs_flat_espresso
import time
import random
from typing import List, Dict, Any, Tuple

from pyeda.inter import exprvars
from espresso_minimization import minimize_truth_table_espresso


def random_tt(n_vars: int, seed: int = 0) -> str:
    rng = random.Random(seed)
    return "".join(rng.choice("01") for _ in range(1 << n_vars))


def build_cofactor_tts(f_tt: str, n_total: int, k_split: int) -> List[str]:
    """
    PyEDA truth-table ordering:
      idx = x[0] + 2*x[1] + ... + 2^{n-1}*x[n-1]
    so x[0] is fast/LSB and x[n-1] is slow/MSB.

    If m = n_total - k_split, then:
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
        cofactors.append(f_tt[start:end])
    return cofactors


def eval_expr_tt(expr, X, m: int) -> List[int]:
    """
    Evaluate expr on all 2^m assignments using PyEDA ordering:
      idx = x[0] + 2*x[1] + ... + 2^{m-1}*x[m-1]
    """
    out = [0] * (1 << m)
    for idx in range(1 << m):
        point = {X[k]: (idx >> k) & 1 for k in range(m)}
        val = expr.restrict(point)
        out[idx] = 1 if val.is_one() else 0
    return out


def hierarchical_espresso_stats(
    f_tt: str,
    n_total: int,
    k_split: int,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Runs Shannon decomposition + Espresso on each cofactor.
    Returns:
      - accuracy (should be 1.0)
      - runtime stats
      - SOP size stats of the *final expanded* representation:
          total_products_final
          total_literals_final
        where selector literals are added to every product term.
    """
    assert len(f_tt) == (1 << n_total)
    assert set(f_tt).issubset({"0", "1"}), "Use only 0/1 for guaranteed 100% accuracy."

    m = n_total - k_split
    num_blocks = 1 << k_split
    block_size = 1 << m

    if verbose:
        print(f"\n=== Hierarchical Espresso (split {k_split} + {m}) ===")
        print(f"num cofactors = 2^{k_split} = {num_blocks}")
        print(f"each cofactor has 2^{m} = {block_size} rows\n")

    # 1) Build cofactors
    t0 = time.perf_counter()
    cofactor_tts = build_cofactor_tts(f_tt, n_total, k_split)
    t_build = time.perf_counter() - t0
    if verbose:
        print(f"[time] build cofactors: {t_build:.3f}s")

    # 2) Espresso each cofactor, collect stats, and precompute predictions
    X_m = exprvars("x", m)
    pred_tables = [None] * num_blocks

    total_espresso_sum = 0.0
    total_terms_cofactors = 0
    total_literals_cofactors = 0

    t0 = time.perf_counter()
    for sel in range(num_blocks):
        expr, stats = minimize_truth_table_espresso(cofactor_tts[sel], verbose=False)
        total_espresso_sum += stats["elapsed"]
        total_terms_cofactors += int(stats.get("num_terms", 0))
        total_literals_cofactors += int(stats.get("total_literals", 0))

        pred_tables[sel] = eval_expr_tt(expr, X_m, m)

    t_espresso_wall = time.perf_counter() - t0
    if verbose:
        print(f"[time] espresso sum elapsed: {total_espresso_sum:.3e}s")
        print(f"[time] espresso wall-clock:  {t_espresso_wall:.3f}s")

    # 3) Exact recombination + accuracy over all 2^n_total
    t0 = time.perf_counter()
    correct = 0
    total = 1 << n_total

    for idx in range(total):
        data = idx & (block_size - 1)  # low bits (x[0]..x[m-1])
        sel = idx >> m                # high bits (x[m]..x[n-1])

        y_hat = pred_tables[sel][data]
        y = 1 if f_tt[idx] == "1" else 0
        correct += (y_hat == y)

    acc = correct / total
    t_acc = time.perf_counter() - t0
    if verbose:
        print(f"[time] accuracy eval over 2^{n_total}: {t_acc:.3f}s")
        print(f"accuracy: {acc:.6f}")

    # 4) Final SOP size after recombination
    # Final expression (expanded SOP form):
    #   OR over sel assignments a:  (selector_a) AND (SOP for cofactor_a)
    #
    # If a cofactor has T terms and L literals, then after AND with selector_a
    # each term gains exactly k_split selector literals.
    #
    # So overall:
    #   total_products_final = sum_a T_a
    #   total_literals_final = sum_a (L_a + k_split * T_a)
    #
    total_products_final = total_terms_cofactors
    total_literals_final = total_literals_cofactors + k_split * total_terms_cofactors

    if verbose:
        if abs(acc - 1.0) < 1e-12:
            print("✅ 100% accuracy (exact decomposition)\n")
        else:
            print("⚠️ Not 100% (ordering mismatch or TT issue)\n")

        print("=== Hierarchical SOP size (expanded) ===")
        print(f"total products (terms): {total_products_final}")
        print(f"total literals:         {total_literals_final}\n")

    return {
        "accuracy": acc,
        "t_build_cofactors": t_build,
        "t_espresso_sum": total_espresso_sum,
        "t_espresso_wall": t_espresso_wall,
        "t_accuracy_eval": t_acc,
        "total_products_final": total_products_final,
        "total_literals_final": total_literals_final,
        "total_products_cofactors": total_terms_cofactors,
        "total_literals_cofactors": total_literals_cofactors,
        "k_split": k_split,
        "m": m,
        "n_total": n_total,
    }


def flat_espresso_stats(
    f_tt: str,
    n_total: int,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run Espresso once on the full n_total-bit truth table.
    Prints and returns accuracy=1.0 by construction (it's minimizing the exact TT),
    plus SOP size stats from Espresso output.
    """
    assert len(f_tt) == (1 << n_total)
    assert set(f_tt).issubset({"0", "1"}), "Use only 0/1 for guaranteed 100% accuracy."

    if verbose:
        print(f"\n=== Flat Espresso ({n_total} bits) ===\n")

    expr, stats = minimize_truth_table_espresso(f_tt, verbose=False)

    # Espresso output itself is an exact representation of the TT:
    # For a fully specified TT, this should match f_tt exactly.
    # We won’t re-evaluate all 2^n points here unless you want.
    num_terms = int(stats.get("num_terms", 0))
    total_literals = int(stats.get("total_literals", 0))

    if verbose:
        print(f"[time] espresso elapsed: {stats['elapsed']:.3e}s")
        print("=== Flat SOP size ===")
        print(f"total products (terms): {num_terms}")
        print(f"total literals:         {total_literals}\n")

    return {
        "t_espresso": stats["elapsed"],
        "total_products": num_terms,
        "total_literals": total_literals,
        "n_total": n_total,
    }


def main():
    # ---- Requested config: 14 bits, 7-7 split ----
    n_total = 14
    k_split = 7  # selector bits count; data bits = n_total - k_split = 7
    seed = 0

    f_tt = random_tt(n_total, seed=seed)

    # Hierarchical (Shannon) + Espresso per cofactor
    hier = hierarchical_espresso_stats(f_tt, n_total=n_total, k_split=k_split, verbose=True)

    # Flat Espresso
    flat = flat_espresso_stats(f_tt, n_total=n_total, verbose=True)

    print("=== Summary ===")
    print(f"Hierarchical accuracy: {hier['accuracy']:.6f}")
    print(f"Hierarchical total products (expanded): {hier['total_products_final']}")
    print(f"Hierarchical total literals (expanded): {hier['total_literals_final']}")
    print(f"Flat total products: {flat['total_products']}")
    print(f"Flat total literals: {flat['total_literals']}")


if __name__ == "__main__":
    main()
