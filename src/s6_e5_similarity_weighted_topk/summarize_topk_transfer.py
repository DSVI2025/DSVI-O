#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))

def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))

def pooled_accuracy_rows(result_root: Path) -> tuple[list[int], list[float], str]:
    pooled_path = result_root / "method2_k_pooled_metrics.csv"
    if pooled_path.exists():
        rows = read_csv(pooled_path)
        rows.sort(key=lambda row: int(row["k"]))
        return (
            [int(row["k"]) for row in rows],
            [float(row["accuracy_percent_pooled"]) for row in rows],
            "pooled_confusion_matrix",
        )

    mean_rows = read_csv(result_root / "target_transfer_k_accuracy_mean.csv")
    mean_rows.sort(key=lambda row: int(row["k"]))
    return (
        [int(row["k"]) for row in mean_rows],
        [float(row["mean_accuracy_percent"]) for row in mean_rows],
        "target_user_mean",
    )

def main() -> None:
    parser = argparse.ArgumentParser(description="Export S6-E5 Top-K transfer table rows.")
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, default=None)
    args = parser.parse_args()

    result_root = args.result_root
    out_root = args.out_root or result_root
    out_root.mkdir(parents=True, exist_ok=True)

    best_k_report = read_json(result_root / "best_k_report.json")
    method2_best = read_json(result_root / "method2_best_k_summary.json")

    k_values, pooled_accuracy, aggregation = pooled_accuracy_rows(result_root)
    latex = (
        "$K$ & "
        + " & ".join(str(value) for value in k_values)
        + " \\\\\nAccuracy (\\%) & "
        + " & ".join(f"{value:.3f}" for value in pooled_accuracy)
        + " \\\\\n"
    )
    (out_root / "table4_transfer_topk_latex_rows.tex").write_text(latex, encoding="utf-8")

    summary = {
        "selected_k": int(method2_best["selected_k"]),
        "selected_k_accuracy_percent": float(method2_best["pooled_metrics"]["accuracy_percent"]),
        "selected_k_macro_f1": float(method2_best["pooled_metrics"]["macro_f1"]),
        "aggregation": aggregation,
        "ranked_candidates": best_k_report["ranked_candidates"],
        "pooled_accuracy_by_k": [
            {"k": int(k), "accuracy_percent": float(acc)}
            for k, acc in zip(k_values, pooled_accuracy)
        ],
        "mean_accuracy_by_k": [
            {"k": int(k), "mean_accuracy_percent": float(acc)}
            for k, acc in zip(k_values, pooled_accuracy)
        ],
    }
    (out_root / "topk_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# S6-E5 Similarity-Weighted Top-K Transfer",
        "",
        f"- selected K: `{summary['selected_k']}`",
        f"- selected-K accuracy: `{summary['selected_k_accuracy_percent']:.6f}%`",
        f"- selected-K macro-F1: `{summary['selected_k_macro_f1']:.6f}`",
        f"- aggregation: `{summary['aggregation']}`",
        "",
        "| K | pooled accuracy (%) |",
        "| ---: | ---: |",
    ]
    for row in summary["pooled_accuracy_by_k"]:
        lines.append(f"| {row['k']} | {row['accuracy_percent']:.3f} |")
    lines.append("")
    (out_root / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
