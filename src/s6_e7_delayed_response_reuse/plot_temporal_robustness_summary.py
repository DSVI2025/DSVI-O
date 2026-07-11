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

def interval_label(minutes: float) -> str:
    if minutes == 0:
        return "0"
    if minutes < 1:
        seconds = int(round(minutes * 60))
        return f"{seconds}s"
    if float(minutes).is_integer():
        return f"{int(minutes)}"
    return f"{minutes:g}"

def cost_reduction_percent(a_steps: int) -> float:
    if a_steps <= 0:
        return 0.0
    return 100.0 * (1.0 - 1.0 / float(a_steps))

def write_stats(path: Path, rows: list[dict[str, float]]) -> None:
    fieldnames = [
        "a_steps",
        "refresh_interval_minutes",
        "cost_reduction_percent",
        "num_users",
        "accuracy_q1",
        "accuracy_median",
        "accuracy_mean",
        "accuracy_q3",
        "accuracy_std",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "a_steps": int(row["a_steps"]),
                    "refresh_interval_minutes": f"{row['refresh_interval_minutes']:.10g}",
                    "cost_reduction_percent": f"{row['cost_reduction_percent']:.10g}",
                    "num_users": int(row["num_users"]),
                    "accuracy_q1": f"{row['accuracy_q1']:.10g}",
                    "accuracy_median": f"{row['accuracy_median']:.10g}",
                    "accuracy_mean": f"{row['accuracy_mean']:.10g}",
                    "accuracy_q3": f"{row['accuracy_q3']:.10g}",
                    "accuracy_std": f"{row['accuracy_std']:.10g}",
                }
            )

def configure_matplotlib() -> None:
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

def plot_accuracy_band(stats_rows: list[dict[str, float]], output_path: Path, dpi: int) -> None:
    positions = np.arange(len(stats_rows))
    med = np.asarray([row["accuracy_median"] for row in stats_rows], dtype=float)
    q1 = np.asarray([row["accuracy_q1"] for row in stats_rows], dtype=float)
    q3 = np.asarray([row["accuracy_q3"] for row in stats_rows], dtype=float)
    labels = [interval_label(row["refresh_interval_minutes"]) for row in stats_rows]

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.fill_between(
        positions,
        q1,
        q3,
        color="#4c78a8",
        alpha=0.22,
        linewidth=0,
        label="25%-75%",
    )
    ax.plot(
        positions,
        med,
        color="#1f4e79",
        marker="o",
        linewidth=2.4,
        markersize=5.5,
        label="median",
    )
    ax.set_title("Temporal Robustness of Response Reuse")
    ax.set_xlabel("Refresh interval (min)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.45)
    ax.legend(frameon=True, framealpha=0.92, loc="lower left")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def plot_tradeoff(stats_rows: list[dict[str, float]], output_path: Path, dpi: int) -> None:
    x = np.asarray([row["cost_reduction_percent"] for row in stats_rows], dtype=float)
    x_plot = -np.log10(np.maximum(100.0 - x, 1e-6))
    mean = np.asarray([row["accuracy_mean"] for row in stats_rows], dtype=float)
    q1 = np.asarray([row["accuracy_q1"] for row in stats_rows], dtype=float)
    q3 = np.asarray([row["accuracy_q3"] for row in stats_rows], dtype=float)
    yerr = np.vstack([mean - q1, q3 - mean])

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.errorbar(
        x_plot,
        mean,
        yerr=yerr,
        color="#1f4e79",
        ecolor="#1f4e79",
        marker="o",
        markersize=5.5,
        linewidth=1.9,
        capsize=3.5,
        elinewidth=1.2,
        label="mean with 25%-75%",
    )
    for idx, row in enumerate(stats_rows):
        label = interval_label(row["refresh_interval_minutes"])
        x_text = -np.log10(max(100.0 - row["cost_reduction_percent"], 1e-6))
        x_offset = 5 if idx < 5 else -24
        y_offset = 5 if idx % 2 == 0 else -12
        ax.annotate(
            label,
            (x_text, row["accuracy_mean"]),
            textcoords="offset points",
            xytext=(x_offset, y_offset),
            fontsize=8,
        )
    ax.set_title("Cost-Accuracy Trade-off")
    ax.set_xlabel("Cost reduction (%)")
    ax.set_ylabel("Mean accuracy (%)")
    tick_values = [0, 80, 90, 95, 98, 99, 99.5, 99.9, 99.99]
    ax.set_xticks([-np.log10(100.0 - value) for value in tick_values])
    ax.set_xticklabels([f"{value:g}" for value in tick_values])
    ax.set_xlim(-2.08, -np.log10(100.0 - 99.995))
    ax.grid(True, axis="both", linestyle="--", linewidth=0.7, alpha=0.45)
    ax.legend(frameon=True, framealpha=0.92, loc="lower left")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot temporal robustness median/IQR and cost-accuracy trade-off."
    )
    parser.add_argument("--metrics-csv", type=Path, required=True)
    parser.add_argument("--accuracy-output", type=Path, required=True)
    parser.add_argument("--tradeoff-output", type=Path, required=True)
    parser.add_argument("--stats-output", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    grouped: dict[int, list[dict[str, float]]] = defaultdict(list)
    for row in read_rows(args.metrics_csv):
        a_steps = int(row["a_steps"])
        if a_steps == 1:
            continue
        grouped[a_steps].append(
            {
                "a_steps": float(a_steps),
                "refresh_interval_minutes": float(row["a_minutes"]),
                "accuracy_percent": float(row["accuracy_percent"]),
            }
        )
    if not grouped:
        raise ValueError(f"No rows found in {args.metrics_csv}")

    stats_rows: list[dict[str, float]] = []
    for a_steps in sorted(grouped):
        rows = grouped[a_steps]
        values = np.asarray([row["accuracy_percent"] for row in rows], dtype=float)
        minutes = float(rows[0]["refresh_interval_minutes"])
        stats_rows.append(
            {
                "a_steps": float(a_steps),
                "refresh_interval_minutes": minutes,
                "cost_reduction_percent": cost_reduction_percent(a_steps),
                "num_users": float(len(values)),
                "accuracy_q1": float(np.percentile(values, 25)),
                "accuracy_median": float(np.median(values)),
                "accuracy_mean": float(values.mean()),
                "accuracy_q3": float(np.percentile(values, 75)),
                "accuracy_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            }
        )

    configure_matplotlib()
    plot_accuracy_band(stats_rows, args.accuracy_output, args.dpi)
    plot_tradeoff(stats_rows, args.tradeoff_output, args.dpi)
    args.stats_output.parent.mkdir(parents=True, exist_ok=True)
    write_stats(args.stats_output, stats_rows)

if __name__ == "__main__":
    main()
