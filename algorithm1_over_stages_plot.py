import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# EDIT ONLY THIS SECTION
# ============================================================

RESULT_FILES = [
    "results_config1/stage_per_seed.csv",
    "results_config2/stage_per_seed.csv",
    "results_config3/stage_per_seed.csv",
    "results_config4/stage_per_seed.csv",
    "results_config5/stage_per_seed.csv",
    "results_config6/stage_per_seed.csv",
    "results_config7/stage_per_seed.csv",
    "results_config8/stage_per_seed.csv",
    "results_config9/stage_per_seed.csv",
    "results_config10/stage_per_seed.csv",
    "results_config11/stage_per_seed.csv",
]

OUTPUT_DIR = "stage_plots"


# ============================================================
# Y-AXIS LIMITS
# ============================================================

Y_LIMITS = {
    "Config. 1":  (0.68, 0.87),
    "Config. 2":  (0.70, 0.88),
    "Config. 3":  (0.68, 0.90),
    "Config. 4":  (0.65, 0.80),
    "Config. 5":  (0.55, 0.65),
    "Config. 6":  (0.65, 0.82),
    "Config. 7":  (0.60, 0.72),
    "Config. 8":  (0.95, 1.00),
    "Config. 9":  (0.54, 0.62),
    "Config. 10": (0.70, 0.90),
    "Config. 11": (0.68, 0.93),
}


# ============================================================
# FONT SETTINGS
# ============================================================

FONT_FAMILY = "Times New Roman"

XLABEL_SIZE = 22
YLABEL_SIZE = 22
TITLE_SIZE = 22
TICK_SIZE = 16

plt.rcParams["font.family"] = FONT_FAMILY


# ============================================================
# CONFIDENCE INTERVAL
# ============================================================

# 95% Student-t critical value for n = 20:
# t_{0.025,19} = 2.093

T_CRIT = 2.093
NUM_SEEDS = 20


# ============================================================
# READ RESULTS
# ============================================================

def read_csv(path):
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


# ============================================================
# MEAN AND 95% CONFIDENCE INTERVAL
# ============================================================

def mean_ci95(values):

    n = len(values)

    if n != NUM_SEEDS:
        raise ValueError(
            f"Expected {NUM_SEEDS} seeds, but got {n}."
        )

    mu = sum(values) / n

    variance = sum(
        (x - mu) ** 2
        for x in values
    ) / (n - 1)

    sd = math.sqrt(variance)

    half_width = (
        T_CRIT
        * sd
        / math.sqrt(n)
    )

    lower = mu - half_width
    upper = mu + half_width

    return mu, lower, upper


# ============================================================
# SAFE FILE NAME
# ============================================================

def safe_filename(name):

    return (
        name
        .replace(" ", "_")
        .replace(".", "")
        .replace("/", "_")
        .replace("\\", "_")
    )


# ============================================================
# MAIN
# ============================================================

def main():

    grouped = defaultdict(
        lambda: defaultdict(list)
    )

    # --------------------------------------------------------
    # Load results
    # --------------------------------------------------------

    for filename in RESULT_FILES:

        path = Path(filename)

        if not path.exists():
            print(f"Skipping missing file: {path}")
            continue

        rows = read_csv(path)

        for row in rows:

            config = row["config"]

            stage = int(
                row["stage"]
            )

            accuracy = float(
                row["accuracy"]
            )

            grouped[config][stage].append(
                accuracy
            )


    if not grouped:
        raise ValueError(
            "No result files were loaded."
        )


    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    output_dir = Path(
        OUTPUT_DIR
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    # --------------------------------------------------------
    # One plot per configuration
    # --------------------------------------------------------

    for config in sorted(grouped):

        stages = sorted(
            grouped[config]
        )

        means = []
        lowers = []
        uppers = []

        for stage in stages:

            values = grouped[
                config
            ][
                stage
            ]

            mu, lo, hi = mean_ci95(
                values
            )

            means.append(
                mu
            )

            lowers.append(
                max(0.0, lo)
            )

            uppers.append(
                min(1.0, hi)
            )


        # ----------------------------------------------------
        # Create figure
        # ----------------------------------------------------

        fig, ax = plt.subplots(
            figsize=(6.5, 4.2)
        )


        # ----------------------------------------------------
        # Mean accuracy curve
        # ----------------------------------------------------

        ax.plot(
            stages,
            means,
            marker="o",
            linewidth=2,
        )


        # ----------------------------------------------------
        # 95% confidence interval
        # ----------------------------------------------------

        ax.fill_between(
            stages,
            lowers,
            uppers,
            alpha=0.18,
        )


        # ----------------------------------------------------
        # Labels and title
        # ----------------------------------------------------

        ax.set_xlabel(
            "Number of residual stages",
            fontsize=XLABEL_SIZE,
            fontname=FONT_FAMILY,
        )

        ax.set_ylabel(
            "Test accuracy",
            fontsize=YLABEL_SIZE,
            fontname=FONT_FAMILY,
        )

        ax.set_title(
            config,
            fontsize=TITLE_SIZE,
            fontname=FONT_FAMILY,
        )


        # ----------------------------------------------------
        # Tick font size
        # ----------------------------------------------------

        ax.tick_params(
            axis="both",
            labelsize=TICK_SIZE,
        )


        # ----------------------------------------------------
        # X-axis
        # ----------------------------------------------------

        ax.set_xlim(
            1,
            max(stages),
        )

        ax.set_xticks(
            range(
                1,
                max(stages) + 1
            )
        )


        # ----------------------------------------------------
        # Configuration-specific y-axis
        # ----------------------------------------------------

        if config in Y_LIMITS:

            ymin, ymax = Y_LIMITS[
                config
            ]

            ax.set_ylim(
                ymin,
                ymax,
            )

        else:

            print(
                f"Warning: no y-limits specified for {config}."
            )


        # ----------------------------------------------------
        # Grid
        # ----------------------------------------------------

        ax.grid(
            True,
            alpha=0.3,
        )


        # ----------------------------------------------------
        # Save figure
        # ----------------------------------------------------

        fig.tight_layout()

        filename = (
            safe_filename(config)
            + "_accuracy_over_stages.pdf"
        )

        output_file = (
            output_dir
            /
            filename
        )

        fig.savefig(
            output_file,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(
            fig
        )

        print(
            f"Saved plot to: {output_file}"
        )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()