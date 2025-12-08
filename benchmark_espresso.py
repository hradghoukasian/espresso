import statistics
from espresso_minimization import generate_random_tt, minimize_truth_table_espresso

def benchmark(ns, runs=10):
    print("\n=== Runtime Benchmark for Espresso (PyEDA) ===\n")
    print(f"{'n':>4} | {'runs':>4} | {'mean time (s)':>15} | {'std (s)':>10}")
    print("-" * 50)

    for n in ns:
        times = []
        for _ in range(runs):
            tt = generate_random_tt(n)
            _, stats = minimize_truth_table_espresso(tt, verbose=False)
            times.append(stats["elapsed"])

        mean_t = statistics.mean(times)
        std_t = statistics.stdev(times) if runs > 1 else 0.0

        print(f"{n:>4} | {runs:>4} | {mean_t:>15.3e} | {std_t:>10.3e}")

    print("\n---------------------------------------------\n")

if __name__ == "__main__":
    sizes = [20]
    benchmark(sizes, runs=1)