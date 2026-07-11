#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))

def percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q))

def write_stats_csv(path: Path, k_values: list[int], grouped_values: list[np.ndarray]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["k", "count", "min", "q1", "median", "mean", "q3", "max"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for k_value, values in zip(k_values, grouped_values):
            writer.writerow(
                {
                    "k": k_value,
                    "count": len(values),
                    "min": f"{values.min():.10g}",
                    "q1": f"{percentile(values, 25):.10g}",
                    "median": f"{np.median(values):.10g}",
                    "mean": f"{values.mean():.10g}",
                    "q3": f"{percentile(values, 75):.10g}",
                    "max": f"{values.max():.10g}",
                }
            )

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot a boxplot summary of target transfer accuracy by K.")
    parser.add_argument("--k-sweep-csv", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--stats-csv", type=Path, default=None)
    parser.add_argument("--title", default="Transfer Accuracy Distribution vs. K")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    rows = read_rows(args.k_sweep_csv)
    by_k: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        by_k[int(row["k"])].append(float(row["accuracy_percent"]))
    if not by_k:
        raise ValueError(f"No rows found in {args.k_sweep_csv}")

    k_values = sorted(by_k)
    grouped_values = [np.asarray(by_k[k], dtype=float) for k in k_values]
    means = np.asarray([values.mean() for values in grouped_values], dtype=float)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "axes.linewidth": 0.9,
        }
    )

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    positions = np.arange(1, len(k_values) + 1)
    box = ax.boxplot(
        grouped_values,
        positions=positions,
        widths=0.58,
        patch_artist=True,
        showfliers=True,
        medianprops={"color": "black", "linewidth": 1.7},
        boxprops={"edgecolor": "#1f4e79", "linewidth": 1.3},
        whiskerprops={"color": "#1f4e79", "linewidth": 1.2},
        capprops={"color": "#1f4e79", "linewidth": 1.2},
        flierprops={
            "marker": "o",
            "markerfacecolor": "white",
            "markeredgecolor": "#1f4e79",
            "markersize": 4.0,
            "alpha": 0.9,
        },
    )
    for patch in box["boxes"]:
        patch.set_facecolor("#d8e8f7")
        patch.set_alpha(0.85)

    ax.plot(
        positions,
        means,
        color="black",
        marker="o",
        linestyle="-",
        linewidth=2.3,
        markersize=5.5,
        label="mean",
        zorder=10,
    )

    ax.set_title(args.title)
    ax.set_xlabel("Number of transferred source users K")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(positions)
    ax.set_xticklabels([str(k) for k in k_values])
    ax.grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.45)
    ax.legend(frameon=True, framealpha=0.92, loc="lower right")

    all_values = np.concatenate(grouped_values)
    value_range = float(all_values.max() - all_values.min())
    margin = max(0.8, value_range * 0.08)
    ax.set_ylim(float(all_values.min() - margin), float(all_values.max() + margin))

    fig.tight_layout()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    if args.stats_csv is not None:
        args.stats_csv.parent.mkdir(parents=True, exist_ok=True)
        write_stats_csv(args.stats_csv, k_values, grouped_values)

if __name__ == "__main__":
    main()
