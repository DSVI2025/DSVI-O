#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

SECONDS_PER_STEP = 5.0
DEFAULT_REFRESH_INTERVALS = "0,6,12,60,120,360,720,2880,8640"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Method 2 transfer accuracy when transferred responses are reused over refresh intervals."
    )
    parser.add_argument(
        "--code-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "s6_e5_similarity_weighted_topk",
        help="Directory containing _transfer_learning_common.py and run_old_matlab_equivalent.py.",
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--response-cache-root", type=Path, required=True)
    parser.add_argument("--pairwise-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--k", type=int, default=8, help="Number of most similar source users to average.")
    parser.add_argument(
        "--refresh-intervals",
        default=None,
        help="Comma-separated nonnegative refresh intervals in time steps. One step is 5 seconds for this dataset.",
    )
    parser.add_argument(
        "--lags",
        default=None,
        help="Deprecated alias for --refresh-intervals.",
    )
    parser.add_argument("--user-ids", default="11,12,13,14,15,16,17,18,19,20")
    parser.add_argument("--storage-dtype", default="float32", choices=["float32", "float64"])
    return parser.parse_args()

def read_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))

def parse_ints(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]

def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def refresh_interval_response(response: np.ndarray, interval_steps: int) -> np.ndarray:
    if interval_steps <= 1:
        return response
    interval = int(interval_steps)
    num_steps = int(response.shape[1])
    refresh_idx = (np.arange(num_steps, dtype=np.int64) // interval) * interval
    return response[:, refresh_idx, :]

def weighted_source_response(
    *,
    common,
    response_cache_root: Path,
    target_user_id: int,
    pairwise_rows: list[dict[str, Any]],
    k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    ranked_rows = sorted(
        [row for row in pairwise_rows if int(row["target_user_id"]) == target_user_id],
        key=lambda row: (float(row["distance"]), int(row["source_user_id"])),
    )
    if len(ranked_rows) < k:
        raise ValueError(f"Target user {target_user_id} has only {len(ranked_rows)} candidate sources; K={k}.")
    selected_rows = ranked_rows[:k]
    distances = np.asarray([float(row["distance"]) for row in selected_rows], dtype=np.float64)
    all_distances = np.asarray([float(row["distance"]) for row in ranked_rows], dtype=np.float64)
    tau = max(float(np.median(all_distances)), 1e-12)
    raw_weights = np.exp(-distances / tau)
    weights = raw_weights / np.sum(raw_weights)

    y1 = None
    y2 = None
    y3 = None
    for weight, row in zip(weights, selected_rows):
        source_user_id = int(row["source_user_id"])
        cache = common.load_response_cache(response_cache_root, source_user_id)
        if not np.array_equal(cache["days"], common.load_runner_module().DAYS.astype(np.int64)):
            raise ValueError(f"Response cache days mismatch for source user {source_user_id}.")
        next_y1 = np.asarray(cache["y1_mean"], dtype=np.float64)
        next_y2 = np.asarray(cache["y2_mean"], dtype=np.float64)
        next_y3 = np.asarray(cache["y3_mean"], dtype=np.float64)
        if y1 is None:
            y1 = weight * next_y1
            y2 = weight * next_y2
            y3 = weight * next_y3
        else:
            y1 += weight * next_y1
            y2 += weight * next_y2
            y3 += weight * next_y3

    assert y1 is not None and y2 is not None and y3 is not None
    meta = {
        "source_user_ids": [int(row["source_user_id"]) for row in selected_rows],
        "source_distances": [float(row["distance"]) for row in selected_rows],
        "weights": [float(value) for value in weights],
        "tau": tau,
    }
    return y1, y2, y3, meta

def plot_accuracy(rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]], output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FixedFormatter, FixedLocator
    except ModuleNotFoundError:
        print("[delayed-refresh] matplotlib is not installed; skipping accuracy-vs-interval plot.", flush=True)
        return

    user_ids = sorted({int(row["target_user_id"]) for row in rows})
    lags = sorted({int(row["a_steps"]) for row in rows})
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(11.6, 6.8), constrained_layout=True)
    cmap = plt.get_cmap("tab10")

    for idx, user_id in enumerate(user_ids):
        user_rows = {int(row["a_steps"]): row for row in rows if int(row["target_user_id"]) == user_id}
        x = [float(user_rows[lag]["a_minutes"]) for lag in lags]
        y = [float(user_rows[lag]["accuracy_percent"]) for lag in lags]
        ax.plot(
            x,
            y,
            marker="o",
            linewidth=1.65,
            markersize=4.8,
            alpha=0.86,
            color=cmap(idx % 10),
            label=f"user {user_id:02d}",
        )

    summary_by_lag = {int(row["a_steps"]): row for row in summary_rows}
    mean_x = [float(summary_by_lag[lag]["a_minutes"]) for lag in lags]
    mean_y = [float(summary_by_lag[lag]["accuracy_mean"]) for lag in lags]
    ax.plot(
        mean_x,
        mean_y,
        marker="s",
        linewidth=3.0,
        markersize=6.0,
        color="black",
        linestyle="--",
        label="mean",
        zorder=8,
    )

    ax.set_xscale("symlog", linthresh=1.0, linscale=0.8)
    ax.xaxis.set_major_locator(FixedLocator(mean_x))
    ax.xaxis.set_major_formatter(FixedFormatter([format_minutes(value) for value in mean_x]))
    ax.tick_params(axis="x", labelrotation=35)
    ax.set_title("Delayed Response Reuse: Accuracy vs. Refresh Interval", fontsize=15, pad=10)
    ax.set_xlabel("Response refresh interval a (minutes, 5-second sampling)")
    ax.set_ylabel("Accuracy (%)")
    ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.38)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(ncol=2, fontsize=9, frameon=True, loc="center left", bbox_to_anchor=(1.02, 0.5))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=240, facecolor="white")
    plt.close(fig)

def format_minutes(value: float) -> str:
    if value == 0:
        return "0"
    if value < 1:
        return f"{value:.2g}"
    if value < 60:
        return f"{value:g}"
    hours = value / 60.0
    if hours < 24:
        return f"{hours:g}h"
    return f"{hours / 24.0:g}d"

def main() -> None:
    args = parse_args()
    if args.k <= 0:
        raise ValueError("--k must be positive.")
    raw_intervals = args.refresh_intervals if args.refresh_intervals is not None else args.lags
    if raw_intervals is None:
        raw_intervals = DEFAULT_REFRESH_INTERVALS
    refresh_intervals = parse_ints(raw_intervals)
    if not refresh_intervals or any(interval < 0 for interval in refresh_intervals):
        raise ValueError("--refresh-intervals must contain nonnegative integers.")
    refresh_intervals = sorted(set(refresh_intervals))
    user_ids = parse_ints(args.user_ids)
    if not user_ids:
        raise ValueError("--user-ids must contain at least one user id.")

    sys.path.insert(0, str(args.code_root))
    import _transfer_learning_common as common  # type: ignore

    runner = common.load_runner_module()
    pairwise_rows = read_json(args.pairwise_json)
    if not isinstance(pairwise_rows, list) or not pairwise_rows:
        raise ValueError("--pairwise-json must contain a non-empty list.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    source_meta: dict[str, Any] = {}
    for user_id in user_ids:
        print(f"[delayed-refresh] target user {user_id:02d}: building K={args.k} weighted response", flush=True)
        y1, y2, y3, meta = weighted_source_response(
            common=common,
            response_cache_root=args.response_cache_root,
            target_user_id=user_id,
            pairwise_rows=pairwise_rows,
            k=args.k,
        )
        source_meta[str(user_id)] = meta
        for interval in refresh_intervals:
            print(f"[delayed-refresh] user {user_id:02d} refresh interval={interval} steps", flush=True)
            summary = common.run_transfer_inference_for_user(
                runner,
                data_dir=args.data_dir,
                user_id=user_id,
                transferred_y1=refresh_interval_response(y1, interval),
                transferred_y2=refresh_interval_response(y2, interval),
                transferred_y3=refresh_interval_response(y3, interval),
                method="method2_similarity_weighted_response_transfer_refresh_interval",
                source_user_ids=meta["source_user_ids"],
                source_distances=meta["source_distances"],
                weights=meta["weights"],
                tau=meta["tau"],
                k=args.k,
                storage_dtype_name=args.storage_dtype,
            )
            rows.append(
                {
                    "target_user_id": user_id,
                    "a_steps": interval,
                    "a_seconds": interval * SECONDS_PER_STEP,
                    "a_minutes": interval * SECONDS_PER_STEP / 60.0,
                    "k": args.k,
                    "accuracy_percent": float(summary["accuracy_percent"]),
                    "macro_f1": float(summary["macro_f1"]),
                    "balanced_accuracy": float(summary["balanced_accuracy"]),
                    "weighted_f1": float(summary["weighted_f1"]),
                    "target_days_runtime_seconds": float(summary["target_days_runtime_seconds"]),
                }
            )

    summary_rows: list[dict[str, Any]] = []
    for interval in refresh_intervals:
        interval_rows = [row for row in rows if int(row["a_steps"]) == interval]
        accuracies = np.asarray([float(row["accuracy_percent"]) for row in interval_rows], dtype=np.float64)
        macro_f1 = np.asarray([float(row["macro_f1"]) for row in interval_rows], dtype=np.float64)
        balanced = np.asarray([float(row["balanced_accuracy"]) for row in interval_rows], dtype=np.float64)
        summary_rows.append(
            {
                "a_steps": interval,
                "a_seconds": interval * SECONDS_PER_STEP,
                "a_minutes": interval * SECONDS_PER_STEP / 60.0,
                "k": args.k,
                "num_users": len(interval_rows),
                "accuracy_mean": float(np.mean(accuracies)),
                "accuracy_std": float(np.std(accuracies, ddof=0)),
                "macro_f1_mean": float(np.mean(macro_f1)),
                "macro_f1_std": float(np.std(macro_f1, ddof=0)),
                "balanced_accuracy_mean": float(np.mean(balanced)),
                "balanced_accuracy_std": float(np.std(balanced, ddof=0)),
            }
        )

    write_csv(
        args.output_dir / "lagged_response_transfer_metrics.csv",
        rows,
        [
            "target_user_id",
            "a_steps",
            "a_seconds",
            "a_minutes",
            "k",
            "accuracy_percent",
            "macro_f1",
            "balanced_accuracy",
            "weighted_f1",
            "target_days_runtime_seconds",
        ],
    )
    write_csv(
        args.output_dir / "lagged_response_transfer_summary.csv",
        summary_rows,
        [
            "a_steps",
            "a_seconds",
            "a_minutes",
            "k",
            "num_users",
            "accuracy_mean",
            "accuracy_std",
            "macro_f1_mean",
            "macro_f1_std",
            "balanced_accuracy_mean",
            "balanced_accuracy_std",
        ],
    )
    (args.output_dir / "lagged_response_transfer_metrics.json").write_text(
        json.dumps(
            {
                "data_dir": str(args.data_dir),
                "response_cache_root": str(args.response_cache_root),
                "pairwise_json": str(args.pairwise_json),
                "k": args.k,
                "refresh_intervals": refresh_intervals,
                "seconds_per_step": SECONDS_PER_STEP,
                "source_meta_by_user": source_meta,
                "rows": rows,
                "summary": summary_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    figure_path = args.output_dir / "lagged_response_accuracy_vs_a.png"
    plot_accuracy(rows, summary_rows, figure_path)
    print(figure_path)
    print(args.output_dir / "lagged_response_transfer_metrics.csv")
    print(args.output_dir / "lagged_response_transfer_summary.csv")

if __name__ == "__main__":
    main()
