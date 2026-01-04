"""
Batch Espresso

A heuristic, hierarchical variant of Espresso minimization.
Inputs are partitioned into batches; each batch is minimized
with Espresso and produces intermediate bits that are fed
into subsequent stages.

This trades exactness for scalability. Accuracy is evaluated
by comparing the composed circuit against the original
n-bit truth table.
"""
# espresso_tree_20bit_demo.py
import time
import random
from typing import List, Tuple

from pyeda.inter import exprvars

from espresso_minimization import minimize_truth_table_espresso


def eval_expr_on_all_inputs(expr, n_vars: int) -> List[int]:
    """
    Evaluate a PyEDA expr over all 2^n_vars assignments.
    Returns a list of 0/1 ints in lexicographic order consistent with index -> bits:
        index = sum_{k=0}^{n-1} bit_k * 2^{n-1-k}
    where variables are x0..x_{n-1}.
    """
    X = exprvars("x", n_vars)
    out = [0] * (1 << n_vars)
    for idx in range(1 << n_vars):
        # x0 is MSB, x_{n-1} is LSB
        point = {X[k]: (idx >> (n_vars - 1 - k)) & 1 for k in range(n_vars)}
        val = expr.restrict(point)
        # val is a PyEDA constant 0/1
        out[idx] = 1 if val.is_one() else 0
    return out


def random_tt(n_vars: int, seed: int = 0) -> str:
    rng = random.Random(seed)
    size = 1 << n_vars
    return "".join(rng.choice("01") for _ in range(size))


def majority_bit(ones: int, zeros: int) -> str:
    if ones > zeros:
        return "1"
    if zeros > ones:
        return "0"
    return "-"  # tie -> don't care


def build_g_tt_from_f(f_tt: str) -> str:
    """
    f is 20-bit: inputs split as (a=first10, b=last10), index = (a<<10)|b.
    Define g(a) = majority_b f(a,b).
    """
    g = ["0"] * (1 << 10)
    for a in range(1 << 10):
        ones = 0
        base = a << 10
        for b in range(1 << 10):
            ones += (f_tt[base | b] == "1")
        zeros = (1 << 10) - ones
        g[a] = majority_bit(ones, zeros)
    return "".join(g)


def build_h_tt_from_f_and_gpred(f_tt: str, g_pred: List[int]) -> str:
    """
    Build an 11-bit truth table for h(g, b) by aggregating all (a,b) that map to same (g_pred[a], b).

    For each z = (g,b) (where g is 1 bit, b is 10 bits):
      h_tt[z] = majority over all a s.t. g_pred[a]=g of f(a,b)
      ties -> '-'
    """
    ones = [0] * (1 << 11)
    zeros = [0] * (1 << 11)

    for a in range(1 << 10):
        ga = g_pred[a]  # 0/1
        base = a << 10
        for b in range(1 << 10):
            z = (ga << 10) | b
            if f_tt[base | b] == "1":
                ones[z] += 1
            else:
                zeros[z] += 1

    h = ["0"] * (1 << 11)
    for z in range(1 << 11):
        h[z] = majority_bit(ones[z], zeros[z])
    return "".join(h)


def accuracy_over_all_inputs(f_tt: str, g_pred_10: List[int], h_pred_11: List[int]) -> float:
    """
    Compute accuracy of f_hat(a,b) = h( g(a), b ) on all 2^20 inputs.
    """
    correct = 0
    total = 1 << 20

    for a in range(1 << 10):
        ga = g_pred_10[a]
        base = a << 10
        for b in range(1 << 10):
            z = (ga << 10) | b
            y_hat = h_pred_11[z]
            y = 1 if f_tt[base | b] == "1" else 0
            correct += (y_hat == y)

    return correct / total


def main(seed: int = 0):
    print("\n=== Espresso Tree Demo: 20 bits -> (10 bits -> 1 bit) -> 11 bits ===\n")

    # 1) Build a random 20-bit function
    t0 = time.perf_counter()
    f_tt = random_tt(20, seed=seed)
    t_build_f = time.perf_counter() - t0
    print(f"[build] f_tt length = {len(f_tt)} (=2^20), time = {t_build_f:.3f}s")

    # 2) Build g truth table (10-bit) by majority over last10
    t0 = time.perf_counter()
    g_tt = build_g_tt_from_f(f_tt)
    t_build_g = time.perf_counter() - t0
    print(f"[build] g_tt length = {len(g_tt)} (=2^10), time = {t_build_g:.3f}s")

    # 3) Espresso on g (10 vars)
    g_expr, g_stats = minimize_truth_table_espresso(g_tt, verbose=False)
    print(f"[espresso] g: n=10 elapsed={g_stats['elapsed']:.3e}s terms={g_stats['num_terms']} lits={g_stats['total_literals']}")

    # Evaluate g_expr on all 10-bit inputs (this is what you actually feed to stage 2)
    t0 = time.perf_counter()
    g_pred = eval_expr_on_all_inputs(g_expr, 10)  # list of 0/1 length 1024
    t_eval_g = time.perf_counter() - t0
    print(f"[eval] g on 2^10 inputs time = {t_eval_g:.3f}s")

    # 4) Build h truth table (11-bit) using g_pred + majority aggregation
    t0 = time.perf_counter()
    h_tt = build_h_tt_from_f_and_gpred(f_tt, g_pred)
    t_build_h = time.perf_counter() - t0
    print(f"[build] h_tt length = {len(h_tt)} (=2^11), time = {t_build_h:.3f}s")

    # 5) Espresso on h (11 vars)
    h_expr, h_stats = minimize_truth_table_espresso(h_tt, verbose=False)
    print(f"[espresso] h: n=11 elapsed={h_stats['elapsed']:.3e}s terms={h_stats['num_terms']} lits={h_stats['total_literals']}")

    # Evaluate h_expr on all 11-bit inputs
    t0 = time.perf_counter()
    h_pred = eval_expr_on_all_inputs(h_expr, 11)  # list length 2048
    t_eval_h = time.perf_counter() - t0
    print(f"[eval] h on 2^11 inputs time = {t_eval_h:.3f}s")

    # 6) Accuracy over all 2^20 inputs
    t0 = time.perf_counter()
    acc = accuracy_over_all_inputs(f_tt, g_pred, h_pred)
    t_acc = time.perf_counter() - t0

    print("\n=== Results ===")
    print(f"accuracy over 2^20 inputs: {acc:.6f}")
    print(f"time for accuracy eval:    {t_acc:.3f}s")

    total_elapsed = (
        t_build_f + t_build_g + g_stats["elapsed"] + t_eval_g +
        t_build_h + h_stats["elapsed"] + t_eval_h + t_acc
    )
    print(f"\n(total approx pipeline time) {total_elapsed:.3f}s\n")


if __name__ == "__main__":
    main(seed=0)
