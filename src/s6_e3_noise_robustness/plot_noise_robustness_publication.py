#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter, FixedLocator

NOISE_ORDER = [
    "additive",
    "drift",
    "multiplicative",
    "cumulative",
    "laplace",
    "impulse",
    "mixed",
]

STYLE_BY_FAMILY = {
    "additive": {"color": "#1f77b4", "linestyle": "-", "marker": "o"},
    "drift": {"color": "#ff7f0e", "linestyle": "--", "marker": "s"},
    "multiplicative": {"color": "#2ca02c", "linestyle": "-.", "marker": "^"},
    "cumulative": {"color": "#d62728", "linestyle": ":", "marker": "D"},
    "laplace": {"color": "#9467bd", "linestyle": (0, (5, 2, 1, 2)), "marker": "v"},
    "impulse": {"color": "#8c564b", "linestyle": (0, (3, 1, 1, 1)), "marker": "P"},
    "mixed": {"color": "#e377c2", "linestyle": (0, (7, 2)), "marker": "X"},
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a publication-style noise robustness figure from aggregate CSV metrics."
    )
    parser.add_argument("--metrics-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-accuracy", type=float, required=True)
    parser.add_argument("--baseline-label", default="clean baseline")
    parser.add_argument("--title", default="Accuracy vs. Noise Variance by Noise Type")
    return parser.parse_args()

def load_rows(path: Path) -> dict[str, dict[float, float]]:
    rows: dict[str, dict[float, float]] = defaultdict(dict)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            family_key = "noise_family" if "noise_family" in row else "noise_type"
            family = row[family_key].strip()
            strength = float(row["strength"])
            accuracy = float(row["accuracy_percent"])
            rows[family][strength] = accuracy
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows

def format_percent(value: float) -> str:
    return f"{value:.5f}".rstrip("0").rstrip(".")

def main() -> None:
    args = parse_args()
    rows = load_rows(args.metrics_csv)
    strengths = sorted({strength for family_rows in rows.values() for strength in family_rows})

    missing = [
        (family, strength)
        for family in NOISE_ORDER
        for strength in strengths
        if family in rows and strength not in rows[family]
    ]
    if missing:
        raise ValueError(f"Missing data points: {missing}")

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11.5,
            "legend.title_fontsize": 12.5,
            "axes.linewidth": 0.9,
            "lines.solid_capstyle": "round",
        }
    )

    fig, ax = plt.subplots(figsize=(9.4, 5.6), constrained_layout=True)
    for family in NOISE_ORDER:
        if family not in rows:
            continue
        style = STYLE_BY_FAMILY[family]
        ax.plot(
            strengths,
            [rows[family][strength] for strength in strengths],
            label=family,
            linewidth=2.2,
            markersize=6.8,
            markeredgewidth=0.95,
            markerfacecolor="white",
            markeredgecolor=style["color"],
            **style,
        )

    ax.axhline(
        args.baseline_accuracy,
        color="0.15",
        linestyle=(0, (5, 2)),
        linewidth=1.8,
        label=f"{args.baseline_label} ({format_percent(args.baseline_accuracy)}%)",
    )
    ax.set_xscale("log")
    ax.set_xlim(min(strengths) * 0.9, max(strengths) * 1.1)
    ax.xaxis.set_major_locator(FixedLocator(strengths))
    ax.xaxis.set_major_formatter(FixedFormatter([f"{strength:g}" for strength in strengths]))
    ax.minorticks_off()
    ax.set_xlabel("Noise variance parameter")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(args.title, pad=11)

    y_values = [accuracy for family_rows in rows.values() for accuracy in family_rows.values()] + [args.baseline_accuracy]
    y_min, y_max = min(y_values), max(y_values)
    y_pad = max((y_max - y_min) * 0.16, 0.16)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.grid(True, which="major", axis="both", color="0.86", linewidth=0.85)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.015, 0.5),
        frameon=True,
        title="Noise type",
        handlelength=2.8,
        borderpad=0.45,
        labelspacing=0.55,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, facecolor="white")
    plt.close(fig)
    print(args.output)

if __name__ == "__main__":
    main()
