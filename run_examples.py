import random
from espresso_minimization import (
    minimize_truth_table_espresso,
    build_tt_from_on_off_sets,
)


def example_random():
    n_vars = 10
    size = 2 ** n_vars
    tt = "".join(random.choice("01-") for _ in range(size))
    print("\n=== RANDOM TRUTH TABLE ===")
    print("Truth table:", tt)
    minimize_truth_table_espresso(tt, verbose=True)


def example_on_off_sets():
    n_vars = 3
    onset = {0, 2, 5}   # 000, 010, 101 => 1
    offset = {1, 7}     # 001, 111 => 0
    tt = build_tt_from_on_off_sets(n_vars, onset, offset)
    print("\n=== FROM ON/OFF SETS ===")
    print("Truth table:", tt)
    minimize_truth_table_espresso(tt, verbose=True)


if __name__ == "__main__":
    example_random()
    example_on_off_sets()


