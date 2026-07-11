#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

DEFAULT_POINTS_PER_BATCH = 172_800

METHOD_LABELS = {
    "baseline_target_specific_round31": "Target-domain full-recomputation baseline",
    "target-domain full-recomputation baseline": "Target-domain full-recomputation baseline",
    "random_source_expected": "Random-source expected",
    "random-source expected": "Random-source expected",
    "method1_nearest_source": "Nearest-source transfer",
    "nearest-source transfer": "Nearest-source transfer",
}

def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))

def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("No rows to write.")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

def method_label(method: str) -> str:
    method_key = method.strip().lower()
    if method_key.startswith("method2_similarity_weighted_best_k_"):
        k_value = method_key.rsplit("_", maxsplit=1)[-1]
        return f"Weighted transfer, K={int(k_value)}"
    return METHOD_LABELS.get(method_key, method.strip())

def parse_float(row: dict[str, str], *keys: str) -> float:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return float(value)
    raise KeyError(f"None of the required columns exists: {keys}")

def runtime_seconds(row: dict[str, str]) -> float:
    if row.get("target_days_runtime_seconds") not in (None, ""):
        return float(row["target_days_runtime_seconds"])
    if row.get("batch_s") not in (None, ""):
        return float(row["batch_s"])
    raise KeyError("Expected target_days_runtime_seconds or batch_s in comparison CSV.")

def baseline_runtime(rows: list[dict[str, Any]]) -> float:
    for row in rows:
        if row["method"] == "Target-domain full-recomputation baseline":
            return float(row["batch_s"])
    raise ValueError("Could not find the target-domain full-recomputation baseline row.")

def format_accuracy(value: float) -> str:
    return f"{value:.3f}"

def format_macro_f1(value: float) -> str:
    return f"{value:.3f}"

def format_point_ms(value: float) -> str:
    if abs(value) < 0.01:
        return f"{value:.5f}"
    return f"{value:.3f}"

def format_batch_seconds(value: float) -> str:
    return f"{value:.3f}"

def format_speedup(value: float) -> str:
    if abs(value - 1.0) < 1e-12:
        return "1.0"
    return f"{value:.1f}"

def latex_method_label(label: str) -> str:
    if label == "Target-domain full-recomputation baseline":
        return r"\shortstack[l]{{Target-domain}\\{full-recomputation baseline}}"
    return label.replace("K=", "$K=") + "$" if "K=" in label else label

def latex_rows(rows: list[dict[str, Any]]) -> str:
    lines = []
    for row in rows:
        lines.append(
            " & ".join(
                [
                    latex_method_label(str(row["method"])),
                    format_accuracy(float(row["accuracy_percent"])),
                    format_macro_f1(float(row["macro_f1"])),
                    format_point_ms(float(row["point_ms"])),
                    format_batch_seconds(float(row["batch_s"])),
                    format_speedup(float(row["speedup"])),
                ]
            )
            + r"\\"
        )
    return "\n".join(lines) + "\n"

def normalize_rows(raw_rows: list[dict[str, str]], points_per_batch: int) -> list[dict[str, Any]]:
    rows = []
    for raw in raw_rows:
        batch_s = runtime_seconds(raw)
        rows.append(
            {
                "method": method_label(raw["method"]),
                "accuracy_percent": parse_float(raw, "accuracy_percent", "Accuracy"),
                "macro_f1": parse_float(raw, "macro_f1", "Macro-F1"),
                "point_ms": batch_s * 1000.0 / float(points_per_batch),
                "batch_s": batch_s,
            }
        )
    base_runtime = baseline_runtime(rows)
    for row in rows:
        row["speedup"] = base_runtime / float(row["batch_s"])
    return rows

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize S6-E6 transfer method latency from S6-E5 method comparison output."
    )
    parser.add_argument("--comparison-csv", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--points-per-batch",
        type=int,
        default=DEFAULT_POINTS_PER_BATCH,
        help="Number of five-second target-grid points represented by one reported batch runtime.",
    )
    args = parser.parse_args()

    if args.points_per_batch <= 0:
        raise ValueError("--points-per-batch must be positive.")

    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = normalize_rows(read_csv(args.comparison_csv), args.points_per_batch)

    write_csv(args.output_root / "table5_method_latency.csv", rows)
    (args.output_root / "table5_method_latency_latex_rows.tex").write_text(
        latex_rows(rows),
        encoding="utf-8",
    )

    summary = {
        "source_comparison_csv": str(args.comparison_csv),
        "points_per_batch": args.points_per_batch,
        "rows": rows,
    }
    (args.output_root / "table5_method_latency_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# S6-E6 Transfer Method Latency",
        "",
        f"- source comparison CSV: `{args.comparison_csv}`",
        f"- points per batch: `{args.points_per_batch}`",
        "",
        "| method | accuracy (%) | macro-F1 | point ms | batch s | speedup |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {format_accuracy(float(row['accuracy_percent']))} | "
            f"{format_macro_f1(float(row['macro_f1']))} | {format_point_ms(float(row['point_ms']))} | "
            f"{format_batch_seconds(float(row['batch_s']))} | {format_speedup(float(row['speedup']))} |"
        )
    lines.append("")
    (args.output_root / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
