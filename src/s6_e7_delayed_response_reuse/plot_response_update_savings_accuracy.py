#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot response-update savings against transfer accuracy from a lagged-response summary CSV."
    )
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--output-png", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument(
        "--drop-five-second-lag",
        action="store_true",
        help="Drop a=1 because it has no update-count saving relative to every-step refresh.",
    )
    return parser.parse_args()

def read_rows(path: Path, *, drop_five_second_lag: bool) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing summary CSV: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            a_steps = int(row["a_steps"])
            if drop_five_second_lag and a_steps == 1:
                continue
            accuracy = float(row["accuracy_mean"])
            macro_f1 = float(row["macro_f1_mean"])
            if a_steps <= 0:
                relative_update_cost = 1.0
                updates_per_hour = 720.0
            else:
                relative_update_cost = 1.0 / float(a_steps)
                updates_per_hour = 720.0 / float(a_steps)
            rows.append(
                {
                    "a_steps": a_steps,
                    "a_minutes": float(row["a_minutes"]),
                    "accuracy_percent": accuracy,
                    "macro_f1": macro_f1,
                    "relative_update_cost_percent": relative_update_cost * 100.0,
                    "updates_saved_percent": (1.0 - relative_update_cost) * 100.0,
                    "updates_per_hour": updates_per_hour,
                }
            )
    return sorted(rows, key=lambda row: (float(row["updates_saved_percent"]), int(row["a_steps"])))

def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "a_steps",
        "a_minutes",
        "updates_per_hour",
        "relative_update_cost_percent",
        "updates_saved_percent",
        "accuracy_percent",
        "macro_f1",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def lag_label(minutes: float) -> str:
    if minutes == 0:
        return "0"
    if minutes < 1:
        return f"{minutes:g}m"
    if minutes < 60:
        return f"{minutes:g}m"
    hours = minutes / 60.0
    if hours < 24:
        return f"{hours:g}h"
    return f"{hours / 24.0:g}d"

def plot(rows: list[dict[str, Any]], output_png: Path) -> None:
    try:
        plot_with_matplotlib(rows, output_png)
    except ModuleNotFoundError:
        plot_with_pillow(rows, output_png)

def plot_with_matplotlib(rows: list[dict[str, Any]], output_png: Path) -> None:
    import matplotlib.pyplot as plt

    x = [float(row["relative_update_cost_percent"]) for row in rows]
    y = [float(row["accuracy_percent"]) for row in rows]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10.8, 6.2), constrained_layout=True)
    ax.plot(x, y, marker="o", linewidth=2.4, markersize=6.5, color="#1f77b4")
    ax.scatter(x, y, s=48, color="#1f77b4", zorder=3)

    for row in rows:
        minutes = float(row["a_minutes"])
        ax.annotate(
            lag_label(minutes),
            (float(row["relative_update_cost_percent"]), float(row["accuracy_percent"])),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8.5,
            color="0.22",
        )

    ax.set_title("Response-Update Cost/Savings vs. Transfer Accuracy", fontsize=15, pad=10)
    ax.set_xlabel("Relative response-update cost (% of every-step refresh, log scale; right = more savings)")
    ax.set_ylabel("Mean accuracy (%)")
    ax.set_xscale("log")
    ax.set_xlim(150.0, 0.008)
    ax.set_xticks([100, 10, 1, 0.1, 0.01])
    ax.set_xticklabels(["100\n0% saved", "10\n90% saved", "1\n99% saved", "0.1\n99.9% saved", "0.01\n99.99% saved"])
    ax.set_ylim(min(y) - 2.5, max(y) + 1.0)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.38)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=240, facecolor="white")
    plt.close(fig)

def plot_with_pillow(rows: list[dict[str, Any]], output_png: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    width, height = 1800, 1040
    margin_left, margin_right = 155, 90
    margin_top, margin_bottom = 105, 175
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    x_left_log = math.log10(100.0)
    x_right_log = math.log10(0.01)
    y_values = [float(row["accuracy_percent"]) for row in rows]
    y_min = min(y_values) - 2.5
    y_max = max(y_values) + 1.0

    def sx_cost(cost_percent: float) -> float:
        value_log = math.log10(max(cost_percent, 0.01))
        return margin_left + (x_left_log - value_log) / (x_left_log - x_right_log) * plot_w

    def sy(value: float) -> float:
        return margin_top + (y_max - value) / (y_max - y_min) * plot_h

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 28)
        small = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 23)
        title_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 36)
    except OSError:
        font = ImageFont.load_default()
        small = ImageFont.load_default()
        title_font = ImageFont.load_default()

    axis = (58, 58, 58)
    grid = (214, 220, 226)
    blue = (31, 119, 180)

    draw.text((margin_left, 34), "Response-Update Cost/Savings vs. Transfer Accuracy", fill=(25, 25, 25), font=title_font)

    x_ticks = [
        (100.0, "100", "0% saved"),
        (10.0, "10", "90% saved"),
        (1.0, "1", "99% saved"),
        (0.1, "0.1", "99.9% saved"),
        (0.01, "0.01", "99.99% saved"),
    ]
    for tick, label, saved in x_ticks:
        x = sx_cost(tick)
        draw.line((x, margin_top, x, margin_top + plot_h), fill=grid, width=1)
        draw.text((x - 26, margin_top + plot_h + 18), label, fill=axis, font=small)
        draw.text((x - 54, margin_top + plot_h + 48), saved, fill=(90, 90, 90), font=small)

    y_start = int((y_min + 4) // 5 * 5)
    for tick in range(y_start, int(y_max) + 1, 5):
        y = sy(float(tick))
        draw.line((margin_left, y, margin_left + plot_w, y), fill=grid, width=1)
        draw.text((margin_left - 82, y - 13), f"{tick}", fill=axis, font=small)

    draw.line((margin_left, margin_top, margin_left, margin_top + plot_h), fill=axis, width=3)
    draw.line((margin_left, margin_top + plot_h, margin_left + plot_w, margin_top + plot_h), fill=axis, width=3)

    points = [
        (sx_cost(float(row["relative_update_cost_percent"])), sy(float(row["accuracy_percent"])), row)
        for row in rows
    ]
    for (x1, y1, _), (x2, y2, _) in zip(points, points[1:]):
        draw.line((x1, y1, x2, y2), fill=blue, width=5)
    for x, y, row in points:
        draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=blue, outline="white", width=2)
        label = lag_label(float(row["a_minutes"]))
        if float(row["a_minutes"]) in {0.0, 0.5, 1.0, 5.0, 10.0}:
            label_x = x + 14
            label_y = y - 35
        else:
            label_x = min(x + 14, width - margin_right - 80)
            label_y = max(y - 16, margin_top + 4)
        draw.text((label_x, label_y), label, fill=(48, 48, 48), font=small)

    x_label = "Relative response-update cost (% of every-step refresh, log scale; right = more savings)"
    draw.text((margin_left + plot_w / 2 - 560, height - 62), x_label, fill=axis, font=font)
    y_label = "Mean accuracy (%)"
    y_text = Image.new("RGBA", (330, 55), (255, 255, 255, 0))
    y_draw = ImageDraw.Draw(y_text)
    y_draw.text((0, 0), y_label, fill=axis, font=font)
    rotated = y_text.rotate(90, expand=True)
    image.paste(rotated, (32, margin_top + plot_h // 2 - rotated.height // 2), rotated)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_png)

def main() -> None:
    args = parse_args()
    rows = read_rows(args.summary_csv, drop_five_second_lag=args.drop_five_second_lag)
    if not rows:
        raise ValueError("No rows available for plotting.")
    if args.output_csv is not None:
        write_csv(args.output_csv, rows)
    plot(rows, args.output_png)
    print(args.output_png)
    if args.output_csv is not None:
        print(args.output_csv)

if __name__ == "__main__":
    main()
