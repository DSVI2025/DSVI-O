from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from _transfer_learning_common import (
    DEFAULTS,
    build_random_source_summary,
    flatten_metric_record,
    load_runner_module,
    write_csv,
    write_json,
)

def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

def mean_target_runtime_seconds(summaries: list[dict[str, object]]) -> float:
    return float(
        np.mean(
            [float(summary["target_days_runtime_seconds"]) for summary in summaries],
            dtype=np.float64,
        )
    )

def pooled_or_mean(row: dict[str, object], pooled_key: str, mean_key: str) -> float:
    if pooled_key in row:
        return float(row[pooled_key])
    return float(row[mean_key])

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", type=Path, default=DEFAULTS.baseline_root)
    parser.add_argument("--transfer-root", type=Path, default=DEFAULTS.transfer_root)
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()

    output_root = args.output_root or args.transfer_root
    output_root.mkdir(parents=True, exist_ok=True)

    runner = load_runner_module()
    baseline_summary = load_json(args.baseline_root / "all_users_summary.json")
    method1_all_pairs = load_json(args.transfer_root / "method1_all_pairs.json")
    method1_nearest = load_json(args.transfer_root / "method1_nearest_summary.json")
    method2_best = load_json(args.transfer_root / "method2_best_k_summary.json")
    best_k_report = load_json(args.transfer_root / "best_k_report.json")

    random_summary = build_random_source_summary(method1_all_pairs)
    write_json(output_root / "random_source_summary.json", random_summary)

    final_rows = [
        {
            "method": "baseline_target_specific_round31",
            "accuracy_percent": float(baseline_summary["pooled_metrics"]["accuracy_percent"]),
            "macro_f1": float(baseline_summary["pooled_metrics"]["macro_f1"]),
            "balanced_accuracy": float(baseline_summary["pooled_metrics"]["balanced_accuracy"]),
            "target_days_runtime_seconds": mean_target_runtime_seconds(baseline_summary["summaries"]),
        },
        {
            "method": "random_source_expected",
            "accuracy_percent": float(random_summary["pooled_metrics"]["accuracy_percent"]),
            "macro_f1": float(random_summary["pooled_metrics"]["macro_f1"]),
            "balanced_accuracy": float(random_summary["pooled_metrics"]["balanced_accuracy"]),
            "target_days_runtime_seconds": float(random_summary["target_days_runtime_seconds_mean"]),
        },
        {
            "method": "method1_nearest_source",
            "accuracy_percent": float(method1_nearest["pooled_metrics"]["accuracy_percent"]),
            "macro_f1": float(method1_nearest["pooled_metrics"]["macro_f1"]),
            "balanced_accuracy": float(method1_nearest["pooled_metrics"]["balanced_accuracy"]),
            "target_days_runtime_seconds": mean_target_runtime_seconds(method1_nearest["summaries"]),
        },
        {
            "method": f"method2_similarity_weighted_best_k_{int(method2_best['selected_k'])}",
            "accuracy_percent": float(method2_best["pooled_metrics"]["accuracy_percent"]),
            "macro_f1": float(method2_best["pooled_metrics"]["macro_f1"]),
            "balanced_accuracy": float(method2_best["pooled_metrics"]["balanced_accuracy"]),
            "target_days_runtime_seconds": mean_target_runtime_seconds(method2_best["summaries"]),
        },
    ]
    write_json(output_root / "final_method_comparison.json", final_rows)
    write_csv(output_root / "final_method_comparison.csv", final_rows)

    lines = [
        "# Transfer Learning Report",
        "",
        f"- baseline_root: `{args.baseline_root}`",
        f"- transfer_root: `{args.transfer_root}`",
        f"- selected_best_k: `{int(method2_best['selected_k'])}`",
        "",
        "## Final Method Comparison",
        "",
        "| method | accuracy_percent | macro_f1 | balanced_accuracy | target_days_runtime_seconds |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in final_rows:
        lines.append(
            f"| `{row['method']}` | `{row['accuracy_percent']:.6f}` | `{row['macro_f1']:.6f}` | "
            f"`{row['balanced_accuracy']:.6f}` | `{row['target_days_runtime_seconds']:.6f}` |"
        )

    lines.extend(
        [
            "",
            "## Method 1 Nearest Source",
            "",
            "| target_user | nearest_source | distance | accuracy_percent | macro_f1 | balanced_accuracy |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for summary in method1_nearest["summaries"]:
        transfer = summary["transfer"]
        lines.append(
            f"| `{int(summary['user_id']):02d}` | `{int(transfer['source_user_ids'][0]):02d}` | "
            f"`{float(transfer['source_distances'][0]):.6f}` | `{float(summary['accuracy_percent']):.6f}` | "
            f"`{float(summary['macro_f1']):.6f}` | `{float(summary['balanced_accuracy']):.6f}` |"
        )

    lines.extend(
        [
            "",
            "## Method 2 Best-K Details",
            "",
            "| target_user | k | source_user_ids | weights | accuracy_percent | macro_f1 | balanced_accuracy |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for summary in method2_best["summaries"]:
        transfer = summary["transfer"]
        source_ids = ",".join(f"{int(value):02d}" for value in transfer["source_user_ids"])
        weights = ",".join(f"{float(value):.4f}" for value in transfer["weights"])
        lines.append(
            f"| `{int(summary['user_id']):02d}` | `{int(transfer['k'])}` | `{source_ids}` | `{weights}` | "
            f"`{float(summary['accuracy_percent']):.6f}` | `{float(summary['macro_f1']):.6f}` | "
            f"`{float(summary['balanced_accuracy']):.6f}` |"
        )

    lines.extend(
        [
            "",
            "## Best-K Ranking",
            "",
            "| k | macro_f1_pooled | balanced_accuracy_pooled | accuracy_percent_pooled | accuracy_percent_mean_user |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in best_k_report["ranked_candidates"]:
        macro_f1 = pooled_or_mean(row, "macro_f1_pooled", "macro_f1_mean")
        balanced_accuracy = pooled_or_mean(
            row, "balanced_accuracy_pooled", "balanced_accuracy_mean"
        )
        accuracy_percent = pooled_or_mean(
            row, "accuracy_percent_pooled", "accuracy_percent_mean"
        )
        accuracy_percent_mean = float(row.get("accuracy_percent_mean", accuracy_percent))
        lines.append(
            f"| `{int(row['k'])}` | `{macro_f1:.6f}` | "
            f"`{balanced_accuracy:.6f}` | `{accuracy_percent:.6f}` | "
            f"`{accuracy_percent_mean:.6f}` |"
        )
    (output_root / "transfer_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

if __name__ == "__main__":
    main()
