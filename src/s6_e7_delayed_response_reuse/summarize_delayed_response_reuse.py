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

def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("No rows to write.")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

def relative_cost_percent(a_steps: int) -> float:
    if a_steps <= 0:
        return 100.0
    return 100.0 / float(a_steps)

def fmt_interval(value: float) -> str:
    if value == 0:
        return "0"
    if float(value).is_integer():
        return str(int(value))
    return f"{value:g}"

def fmt_cost(value: float) -> str:
    return f"{value:.3f}"

def fmt_accuracy(value: float) -> str:
    return f"{value:.3f}"

def build_rows(summary_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in sorted(summary_rows, key=lambda item: int(item["a_steps"])):
        a_steps = int(row["a_steps"])
        rows.append(
            {
                "refresh_interval_min": float(row["a_minutes"]),
                "a_steps": a_steps,
                "relative_cost_percent": relative_cost_percent(a_steps),
                "mean_accuracy_percent": float(row["accuracy_mean"]),
                "macro_f1": float(row["macro_f1_mean"]),
                "balanced_accuracy": float(row["balanced_accuracy_mean"]),
                "num_users": int(row["num_users"]),
            }
        )
    return rows

def latex_rows(rows: list[dict[str, Any]]) -> str:
    intervals = " & ".join(fmt_interval(float(row["refresh_interval_min"])) for row in rows)
    costs = " & ".join(fmt_cost(float(row["relative_cost_percent"])) for row in rows)
    accuracy = " & ".join(fmt_accuracy(float(row["mean_accuracy_percent"])) for row in rows)
    return (
        f"Refresh interval (min) & {intervals} \\\\\n"
        f"Relative cost (\\%) & {costs} \\\\\n"
        f"Mean accuracy (\\%) & {accuracy} \\\\\n"
    )

def main() -> None:
    parser = argparse.ArgumentParser(description="Export S6-E7 delayed response reuse table rows.")
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = build_rows(read_csv(args.summary_csv))

    write_csv(args.output_root / "table6_delayed_response_reuse.csv", rows)
    (args.output_root / "table6_delayed_response_reuse_latex_rows.tex").write_text(
        latex_rows(rows),
        encoding="utf-8",
    )

    summary = {
        "source_summary_csv": str(args.summary_csv),
        "rows": rows,
    }
    (args.output_root / "table6_delayed_response_reuse_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# S6-E7 Delayed Response Reuse",
        "",
        f"- source summary CSV: `{args.summary_csv}`",
        "",
        "| refresh interval (min) | relative cost (%) | mean accuracy (%) | macro-F1 |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {fmt_interval(float(row['refresh_interval_min']))} | "
            f"{fmt_cost(float(row['relative_cost_percent']))} | "
            f"{fmt_accuracy(float(row['mean_accuracy_percent']))} | "
            f"{float(row['macro_f1']):.6f} |"
        )
    lines.append("")
    (args.output_root / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
