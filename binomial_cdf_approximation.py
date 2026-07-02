# P_good #1
#
# import math
# import numpy as np
# import matplotlib
#
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
#
#
# def p_good(N: int, B: int) -> float:
#     if N < 1 or B < 1:
#         raise ValueError("N and B must be positive integers.")
#
#     return (N - 1) / (2 ** B - 1)
#
#
# def binomial_cdf_at_most_G(N: int, G: int, p: float) -> float:
#     """
#     Stable computation of Pr(X <= G) for X ~ Binomial(N, p)
#     without using math.comb.
#     """
#     if G < 0:
#         return 0.0
#     if G >= N:
#         return 1.0
#     if not (0.0 <= p <= 1.0):
#         raise ValueError("p must be in [0,1].")
#
#     if p == 0.0:
#         return 1.0
#     if p == 1.0:
#         return 0.0 if G < N else 1.0
#
#     # Start from pmf(0) = (1-p)^N
#     pmf = (1.0 - p) ** N
#     cdf = pmf
#
#     # Recurrence:
#     # pmf(g+1) = pmf(g) * ((N-g)/(g+1)) * (p/(1-p))
#     for g in range(0, G):
#         pmf *= ((N - g) / (g + 1)) * (p / (1.0 - p))
#         cdf += pmf
#
#     return min(max(cdf, 0.0), 1.0)
#
#
# def compute_cdf_curve(N: int, B: int):
#     p = p_good(N, B)
#     G_values = np.arange(0, N + 1)
#     cdf_values = np.array([binomial_cdf_at_most_G(N, int(G), p) for G in G_values])
#     return G_values, cdf_values, p
#
#
# def plot_cdf(N: int, B: int,  save_path: str = "cdf_good_points.png"):
#     G_values, cdf_values, p = compute_cdf_curve(N, B)
#
#     plt.figure(figsize=(8, 5))
#     plt.step(G_values, cdf_values, where="post", label=r"$\Pr(G_i \leq G)$")
#     plt.xlabel("G")
#     plt.ylabel("CDF")
#     plt.title(
#         rf"CDF of $G_i \sim \mathrm{{Binomial}}(N, p_{{good}})$"
#         + "\n"
#         + rf"$N={N},\ B={B},\ p_{{good}}={p:.4e}$"
#     )
#     plt.ylim(-0.02, 1.02)
#     plt.grid(True, alpha=0.3)
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=300, bbox_inches="tight")
#     plt.close()
#     print(f"Figure saved to: {save_path}")
#
#
# if __name__ == "__main__":
#     N = 500
#     B = 15
#
#     p = p_good(N, B)
#     print(f"p_good = {p:.3e}")
#
#     for G in [440,450,450,470,480]:
#         prob = binomial_cdf_at_most_G(N, G, p)
#         print(f"Pr(G_i <= {G}) = {prob:.6f}")
#
#     plot_cdf(N, B,  save_path="cdf_good_points.png")

## P_good #2

import math
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def p_good(N: int, B: int, S: int) -> float:
    """
    p_good = Pr(exists i in J such that x^{⊕ i} in D_train | x in D_train)
           = 1 - C(2^B - 1 - S, N - 1) / C(2^B - 1, N - 1)
    where S = |J|.
    """
    if N < 1 or B < 1 or S < 0:
        raise ValueError("N and B must be positive integers, and S must be nonnegative.")

    M = 2 ** B - 1  # number of points other than x

    if S > B:
        raise ValueError("S cannot exceed B.")
    if N > 2 ** B:
        raise ValueError("Cannot sample more than 2^B distinct points.")
    if N == 1 or S == 0:
        return 0.0

    # If we select more than M-S points, we must hit at least one of the S neighbors
    if N - 1 > M - S:
        return 1.0

    # Stable computation using log-gamma
    log_num = math.lgamma(M - S + 1) - math.lgamma(N) - math.lgamma(M - S - (N - 1) + 1)
    log_den = math.lgamma(M + 1) - math.lgamma(N) - math.lgamma(M - (N - 1) + 1)

    ratio = math.exp(log_num - log_den)
    return min(max(1.0 - ratio, 0.0), 1.0)


def binomial_cdf_at_most_G(N: int, G: int, p: float) -> float:
    """
    Numerically stable normal approximation with continuity correction:
        Pr(X <= G),  X ~ Binomial(N, p)
    """
    if G < 0:
        return 0.0
    if G >= N:
        return 1.0
    if not (0.0 <= p <= 1.0):
        raise ValueError("p must be in [0,1].")

    mu = N * p
    var = N * p * (1.0 - p)

    if var == 0.0:
        return 1.0 if G >= mu else 0.0

    sigma = math.sqrt(var)
    z = (G + 0.5 - mu) / sigma
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def compute_cdf_curve(N: int, B: int, S: int, zoom: bool = False):
    p = p_good(N, B, S)

    if zoom:
        mu = N * p
        sigma = math.sqrt(N * p * (1.0 - p))
        lo = max(0, int(math.floor(mu - 5 * sigma)))
        hi = min(N, int(math.ceil(mu + 5 * sigma)))
        G_values = np.arange(lo, hi + 1)
    else:
        G_values = np.arange(0, N + 1)

    cdf_values = np.array([binomial_cdf_at_most_G(N, int(G), p) for G in G_values])
    return G_values, cdf_values, p


def plot_cdf(N: int, B: int, S: int, save_path: str = "cdf_good_points.png", zoom: bool = False):
    G_values, cdf_values, p = compute_cdf_curve(N, B, S, zoom=zoom)

    plt.figure(figsize=(8, 5))
    plt.step(G_values, cdf_values, where="post", label=r"$\Pr(G \leq g)$")
    plt.xlabel("g")
    plt.ylabel("CDF")
    plt.title(
        rf"CDF of $G \sim \mathrm{{Binomial}}(N, p_{{good}})$"
        + "\n"
        + rf"$N={N},\ B={B},\ S={S},\ p_{{good}}={p:.4e}$"
    )
    plt.ylim(-0.02, 1.02)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Figure saved to: {save_path}")


if __name__ == "__main__":
    N = 500
    B = 15
    S = 8

    p = p_good(N, B, S)
    print(f"p_good = {p:.6e}")

    mu = N * p
    sigma = math.sqrt(N * p * (1.0 - p))
    print(f"mean = {mu:.3f}, std = {sigma:.3f}")

    for G in [2400, 2500, 2588, 2600, 2700]:
        prob = binomial_cdf_at_most_G(N, G, p)
        print(f"Pr(G <= {G}) = {prob:.6f}")

    # Full-range plot
    plot_cdf(N, B, S, save_path="cdf_good_points_full.png", zoom=False)

    # Zoomed plot around the transition region
    plot_cdf(N, B, S, save_path="cdf_good_points_zoom.png", zoom=True)