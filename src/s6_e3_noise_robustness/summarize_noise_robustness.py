#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

TABLE_ORDER = [
    ("cumulative", "Cumulative"),
    ("laplace", "Laplace"),
    ("impulse", "Impulse"),
    ("multiplicative", "Multiplicative"),
    ("additive", "Additive"),
    ("drift", "Drift"),
    ("mixed", "Mixed"),
]

CURVE_ORDER = [
    "additive",
    "drift",
    "multiplicative",
    "cumulative",
    "laplace",
    "impulse",
    "mixed",
]

TABLE_STRENGTHS = [5.0, 10.0, 20.0]

def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))

def load_accuracy(summary_path: Path) -> float:
    summary = read_json(summary_path)
    pooled = summary.get("pooled_metrics")
    if not isinstance(pooled, dict) or "accuracy_percent" not in pooled:
        raise ValueError(f"Missing pooled_metrics.accuracy_percent in {summary_path}")
    return float(pooled["accuracy_percent"])

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
            f"{row['noise_family']} & "
            f"{float(row['s_5_accuracy_percent']):.3f} & "
            f"{float(row['s_10_accuracy_percent']):.3f} & "
            f"{float(row['s_20_accuracy_percent']):.3f} \\\\"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def write_markdown(
    path: Path,
    baseline_accuracy: float,
    table_rows: list[dict[str, object]],
    curve_rows: list[dict[str, object]],
) -> None:
    lines = [
        "# S6-E3 Noise Robustness",
        "",
        f"Clean baseline accuracy: `{baseline_accuracy:.6f}%`.",
        "",
        "## Table 3 Rows",
        "",
        "| Noise family | s=5 | s=10 | s=20 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in table_rows:
        lines.append(
            f"| {row['noise_family']} | "
            f"{float(row['s_5_accuracy_percent']):.3f} | "
            f"{float(row['s_10_accuracy_percent']):.3f} | "
            f"{float(row['s_20_accuracy_percent']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Full Curve Rows",
            "",
            "| noise_type | strength | accuracy (%) |",
            "| --- | ---: | ---: |",
        ]
    )
    for row in curve_rows:
        lines.append(
            f"| {row['noise_type']} | "
            f"{float(row['strength']):g} | "
            f"{float(row['accuracy_percent']):.6f} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export S6-E3 noise robustness table and curve metrics from suite_results.json."
    )
    parser.add_argument("--suite-results", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, default=None)
    args = parser.parse_args()

    suite_results = read_json(args.suite_results)
    results = suite_results.get("results")
    if not isinstance(results, list):
        raise ValueError(f"{args.suite_results} must contain a list field named results")

    baseline_accuracy: float | None = None
    accuracy_by_family_strength: dict[tuple[str, float], float] = {}
    for result in results:
        if result.get("status") != "ok":
            raise ValueError(f"Non-ok experiment in suite results: {result.get('name')}")
        summary_path_raw = result.get("summary_path")
        if not summary_path_raw:
            continue
        family = str(result.get("noise_component_mode"))
        strength = float(result.get("noise_strength_multiplier"))
        accuracy = load_accuracy(Path(summary_path_raw))
        if family == "none":
            baseline_accuracy = accuracy
        else:
            accuracy_by_family_strength[(family, strength)] = accuracy

    if baseline_accuracy is None:
        raise ValueError("No clean baseline result found in suite_results.json")
    if not accuracy_by_family_strength:
        raise ValueError("No noisy experiment results found in suite_results.json")

    observed_strengths = sorted({strength for _, strength in accuracy_by_family_strength})
    missing_table_points = [
        (family, strength)
        for family, _label in TABLE_ORDER
        for strength in TABLE_STRENGTHS
        if (family, strength) not in accuracy_by_family_strength
    ]
    if missing_table_points:
        raise ValueError(f"Missing Table 3 points: {missing_table_points}")

    table_rows: list[dict[str, object]] = []
    for family, label in TABLE_ORDER:
        table_rows.append(
            {
                "noise_family": label,
                "s_5_accuracy_percent": accuracy_by_family_strength[(family, 5.0)],
                "s_10_accuracy_percent": accuracy_by_family_strength[(family, 10.0)],
                "s_20_accuracy_percent": accuracy_by_family_strength[(family, 20.0)],
            }
        )

    curve_rows: list[dict[str, object]] = []
    for family in CURVE_ORDER:
        for strength in observed_strengths:
            key = (family, strength)
            if key in accuracy_by_family_strength:
                curve_rows.append(
                    {
                        "noise_type": family,
                        "strength": strength,
                        "accuracy_percent": accuracy_by_family_strength[key],
                    }
                )

    out_root = args.out_root or args.suite_results.parent
    out_root.mkdir(parents=True, exist_ok=True)
    write_csv(out_root / "noise_accuracy_by_type_vs_strength.csv", curve_rows)
    write_csv(out_root / "table3_noise_robustness.csv", table_rows)
    write_latex_rows(out_root / "table3_noise_robustness_latex_rows.tex", table_rows)
    write_markdown(out_root / "summary.md", baseline_accuracy, table_rows, curve_rows)
    payload = {
        "baseline_accuracy_percent": baseline_accuracy,
        "table_rows": table_rows,
        "curve_rows": curve_rows,
    }
    (out_root / "noise_robustness_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
