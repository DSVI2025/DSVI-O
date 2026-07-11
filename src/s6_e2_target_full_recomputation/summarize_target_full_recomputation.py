#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

CLASS_NAMES = ["healthy (0)", "weak (1)", "ill (2)"]

def class_accuracy_from_confusion(confusion: np.ndarray) -> list[float]:
    total = float(confusion.sum())
    values: list[float] = []
    for idx in range(len(CLASS_NAMES)):
        tp = float(confusion[idx, idx])
        fp = float(confusion[:, idx].sum() - confusion[idx, idx])
        fn = float(confusion[idx, :].sum() - confusion[idx, idx])
        tn = total - tp - fp - fn
        values.append((tp + tn) / max(total, 1.0))
    return values

def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

def write_latex_rows(path: Path, rows: list[dict[str, object]]) -> None:
    lines = []
    for row in rows:
        lines.append(
            f"{row['class']} & "
            f"{float(row['accuracy']):.4f} & "
            f"{float(row['precision']):.4f} & "
            f"{float(row['recall']):.4f} & "
            f"{float(row['specificity']):.4f} & "
            f"{float(row['f1']):.4f} \\\\"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def write_markdown(path: Path, method_row: dict[str, object], class_rows: list[dict[str, object]]) -> None:
    lines = [
        "# S6-E2 Target-Domain Full-Recomputation Baseline",
        "",
        "| accuracy (%) | macro-F1 | balanced accuracy | weighted F1 | worst recall | worst F1 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| {float(method_row['accuracy_percent']):.6f} | "
            f"{float(method_row['macro_f1']):.6f} | "
            f"{float(method_row['balanced_accuracy']):.6f} | "
            f"{float(method_row['weighted_f1']):.6f} | "
            f"{float(method_row['worst_class_recall']):.6f} | "
            f"{float(method_row['worst_class_f1']):.6f} |"
        ),
        "",
        "| class | accuracy | precision | recall | specificity | F1 | support |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in class_rows:
        lines.append(
            f"| {row['class']} | "
            f"{float(row['accuracy']):.4f} | "
            f"{float(row['precision']):.4f} | "
            f"{float(row['recall']):.4f} | "
            f"{float(row['specificity']):.4f} | "
            f"{float(row['f1']):.4f} | "
            f"{int(row['support'])} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export S6-E2 target full-recomputation metrics from all_users_summary.json."
    )
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, default=None)
    args = parser.parse_args()

    result_root = args.result_root
    out_root = args.out_root or result_root
    summary_path = result_root / "all_users_summary.json"
    aggregate = json.loads(summary_path.read_text(encoding="utf-8"))
    pooled = aggregate["pooled_metrics"]
    confusion = np.asarray(aggregate["pooled_confusion_matrix"], dtype=np.int64)
    class_accuracy = class_accuracy_from_confusion(confusion)

    class_rows: list[dict[str, object]] = []
    for idx, class_name in enumerate(CLASS_NAMES):
        class_rows.append(
            {
                "class": class_name,
                "class_index": idx,
                "accuracy": class_accuracy[idx],
                "precision": pooled["precision"][idx],
                "recall": pooled["recall"][idx],
                "specificity": pooled["specificity"][idx],
                "f1": pooled["f1"][idx],
                "support": pooled["support"][idx],
            }
        )

    method_row = {
        "method": "Target-domain full-recomputation baseline",
        "accuracy_percent": pooled["accuracy_percent"],
        "macro_precision": pooled["macro_precision"],
        "macro_recall": pooled["macro_recall"],
        "macro_f1": pooled["macro_f1"],
        "balanced_accuracy": pooled["balanced_accuracy"],
        "weighted_f1": pooled["weighted_f1"],
        "worst_class_recall": pooled["worst_class_recall"],
        "worst_class_f1": pooled["worst_class_f1"],
        "result_root": str(result_root),
    }

    out_root.mkdir(parents=True, exist_ok=True)
    write_csv(out_root / "table2_target_full_recomputation.csv", class_rows)
    write_csv(out_root / "target_full_recomputation_method_summary.csv", [method_row])
    write_latex_rows(out_root / "table2_target_full_recomputation_latex_rows.tex", class_rows)
    write_markdown(out_root / "summary.md", method_row, class_rows)
    print(json.dumps({"method_summary": method_row, "class_rows": class_rows}, indent=2))

if __name__ == "__main__":
    main()
