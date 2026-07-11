#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]
LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]

def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))

def write_mean_csv(path: Path, k_values: list[int], means: list[float]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["k", "mean_accuracy_percent"])
        writer.writeheader()
        for k_value, mean_value in zip(k_values, means):
            writer.writerow({"k": k_value, "mean_accuracy_percent": f"{mean_value:.10g}"})

def read_pooled_accuracy(path: Path, k_values: list[int]) -> np.ndarray:
    rows = read_rows(path)
    by_k: dict[int, float] = {}
    for row in rows:
        k_value = int(row["k"])
        if "accuracy_percent_pooled" in row and row["accuracy_percent_pooled"] != "":
            by_k[k_value] = float(row["accuracy_percent_pooled"])
        elif "accuracy_percent" in row and row["accuracy_percent"] != "":
            by_k[k_value] = float(row["accuracy_percent"])
        elif "mean_accuracy_percent" in row and row["mean_accuracy_percent"] != "":
            by_k[k_value] = float(row["mean_accuracy_percent"])
    missing = [k_value for k_value in k_values if k_value not in by_k]
    if missing:
        raise ValueError(f"Missing pooled accuracy for K values {missing} in {path}")
    return np.asarray([by_k[k_value] for k_value in k_values], dtype=float)

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot target transfer accuracy over top-K source count.")
    parser.add_argument("--k-sweep-csv", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--mean-csv", type=Path, default=None)
    parser.add_argument("--pooled-metrics-csv", type=Path, default=None)
    parser.add_argument("--title", default="Transfer Accuracy vs. Number of Source Users")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    rows = read_rows(args.k_sweep_csv)
    by_user: dict[int, dict[int, float]] = defaultdict(dict)
    for row in rows:
        by_user[int(row["user_id"])][int(row["k"])] = float(row["accuracy_percent"])
    if not by_user:
        raise ValueError(f"No rows found in {args.k_sweep_csv}")

    user_ids = sorted(by_user)
    k_values = sorted({k for values in by_user.values() for k in values})
    matrix = np.asarray([[by_user[user_id][k] for k in k_values] for user_id in user_ids], dtype=float)
    if args.pooled_metrics_csv is not None and args.pooled_metrics_csv.exists():
        means = read_pooled_accuracy(args.pooled_metrics_csv, k_values)
        aggregate_label = "pooled"
    else:
        means = matrix.mean(axis=0)
        aggregate_label = "mean"

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
    fig_height = 4.9 if len(user_ids) > 6 else 4.4
    fig, ax = plt.subplots(figsize=(7.0, fig_height))
    cmap = plt.get_cmap("tab10")
    for idx, user_id in enumerate(user_ids):
        ax.plot(
            k_values,
            matrix[idx],
            color=cmap(idx % 10),
            marker=MARKERS[idx % len(MARKERS)],
            linestyle=LINESTYLES[idx % len(LINESTYLES)],
            linewidth=1.6,
            markersize=5.5,
            label=f"target {user_id}",
        )

    ax.plot(
        k_values,
        means,
        color="black",
        marker="o",
        linestyle="-",
        linewidth=3.0,
        markersize=6.5,
        label=aggregate_label,
        zorder=10,
    )
    ax.set_title(args.title)
    ax.set_xlabel("Number of transferred source users K")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(k_values)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.45)
    if len(user_ids) > 6:
        ax.legend(
            ncol=3,
            frameon=True,
            framealpha=0.92,
            loc="lower right",
            columnspacing=0.9,
            handlelength=2.1,
        )
        fig.tight_layout()
    else:
        ax.legend(ncol=2, frameon=True)
        fig.tight_layout()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    if args.mean_csv is not None:
        args.mean_csv.parent.mkdir(parents=True, exist_ok=True)
        write_mean_csv(args.mean_csv, k_values, means.tolist())

if __name__ == "__main__":
    main()
