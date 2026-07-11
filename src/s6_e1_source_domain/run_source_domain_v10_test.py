#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

DEFAULT_CODE_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "mat" / "source"
DEFAULT_OUT_ROOT = Path(__file__).resolve().parents[2] / "results" / "s6_e1_source_domain"

def parse_int_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]

def load_runner(code_root: Path):
    runner_path = code_root / "run_old_matlab_equivalent.py"
    spec = importlib.util.spec_from_file_location("source_domain_v10_runner", runner_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load runner from {runner_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

def write_confusion_csv(path: Path, matrix: list[list[int]]) -> None:
    path.write_text(
        "true,pred_0,pred_1,pred_2\n"
        + "\n".join(
            f"{idx},{row[0]},{row[1]},{row[2]}" for idx, row in enumerate(matrix)
        )
        + "\n",
        encoding="utf-8",
    )

def write_table1_csv(path: Path, aggregate: dict[str, object]) -> None:
    pooled_metrics = aggregate["pooled_metrics"]
    confusion = np.asarray(aggregate["pooled_confusion_matrix"], dtype=np.int64)
    total = int(confusion.sum())
    precision = pooled_metrics["precision"]
    recall = pooled_metrics["recall"]
    specificity = pooled_metrics["specificity"]
    f1 = pooled_metrics["f1"]
    support = pooled_metrics["support"]
    rows = ["class,accuracy,precision,recall,specificity,f1,support"]
    labels = ["healthy_0", "weak_1", "ill_2"]
    for idx, label in enumerate(labels):
        tp = int(confusion[idx, idx])
        fn = int(confusion[idx, :].sum() - tp)
        fp = int(confusion[:, idx].sum() - tp)
        tn = total - tp - fn - fp
        accuracy = (tp + tn) / total if total else 0.0
        rows.append(
            f"{label},{accuracy:.10f},{precision[idx]:.10f},{recall[idx]:.10f},"
            f"{specificity[idx]:.10f},{f1[idx]:.10f},{support[idx]}"
        )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")

def write_readme(path: Path, aggregate: dict[str, object], args: argparse.Namespace) -> None:
    pooled = aggregate["pooled_metrics"]
    lines = [
        "# Source-Domain v1-v9 Train, v10 Test",
        "",
        "This run evaluates source users with day rows 91-100 held out, matching the interpretation that v1-v9 provide 90 reference days and v10 provides 10 evaluation days.",
        "",
        "## Split",
        "",
        f"- users: `{args.user_ids}`",
        f"- train/reference days: `{args.reference_days}`",
        f"- test/evaluation days: `{args.test_days}`",
        f"- reference sampling: `{args.reference_sampling_mode}:{args.reference_sample_size}x{args.reference_num_trials}`",
        f"- reference seed: `{args.reference_sampling_seed}`",
        f"- device: `{args.device}`",
        f"- torch dtype: `{args.torch_dtype}`",
        f"- storage dtype: `{args.storage_dtype}`",
        "",
        "## Pooled Metrics",
        "",
        f"- accuracy_percent: `{pooled['accuracy_percent']:.6f}`",
        f"- macro_f1: `{pooled['macro_f1']:.6f}`",
        f"- balanced_accuracy: `{pooled['balanced_accuracy']:.6f}`",
        "",
        "## Class Metrics",
        "",
        "| class | accuracy | precision | recall | specificity | f1 | support |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    labels = ["healthy (0)", "weak (1)", "ill (2)"]
    confusion = np.asarray(aggregate["pooled_confusion_matrix"], dtype=np.int64)
    total = int(confusion.sum())
    for idx, label in enumerate(labels):
        tp = int(confusion[idx, idx])
        fn = int(confusion[idx, :].sum() - tp)
        fp = int(confusion[:, idx].sum() - tp)
        tn = total - tp - fn - fp
        accuracy = (tp + tn) / total if total else 0.0
        lines.append(
            f"| {label} | {accuracy:.6f} | {pooled['precision'][idx]:.6f} | {pooled['recall'][idx]:.6f} | "
            f"{pooled['specificity'][idx]:.6f} | {pooled['f1'][idx]:.6f} | {pooled['support'][idx]} |"
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `all_users_summary.json`: full aggregate and per-user summaries",
            "- `pooled_confusion_matrix.csv`: pooled 3-class confusion matrix",
            "- `table1_source_domain_metrics.csv`: class-wise metrics for manuscript table comparison",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run source-domain evaluation with v1-v9 as reference days and v10 as held-out days."
    )
    parser.add_argument("--code-root", type=Path, default=DEFAULT_CODE_ROOT)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--user-ids", default="1,2,3,4,5,6,7,8,9,10")
    parser.add_argument("--reference-days", default="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90")
    parser.add_argument("--test-days", default="91,92,93,94,95,96,97,98,99,100")
    parser.add_argument("--reference-sampling-mode", choices=["sampled", "all"], default="sampled")
    parser.add_argument("--reference-sample-size", type=int, default=50)
    parser.add_argument("--reference-num-trials", type=int, default=20)
    parser.add_argument("--reference-sampling-seed", type=int, default=20260423)
    parser.add_argument("--history-window", type=int, default=12)
    parser.add_argument("--storage-dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--target-day-batch-size", type=int, default=10)
    parser.add_argument("--history-chunk-size", type=int, default=1024)
    parser.add_argument("--progress-interval", type=int, default=0)
    parser.add_argument("--params-hat-rho", type=float, default=0.7)
    parser.add_argument("--params-hat-lambda", type=float, default=0.305)
    parser.add_argument("--params-hat-alpha-mix", default="0.75,0.58,0.16")
    parser.add_argument("--verify-against-cpu", action="store_true")
    args = parser.parse_args()

    user_ids = parse_int_list(args.user_ids)
    reference_days = parse_int_list(args.reference_days)
    test_days = parse_int_list(args.test_days)
    if sorted(reference_days + test_days) != list(range(1, 101)):
        raise ValueError("reference-days and test-days must partition 1..100.")
    if set(reference_days) & set(test_days):
        raise ValueError("reference-days and test-days overlap.")

    runner = load_runner(args.code_root)
    runner.DAYS = np.asarray(test_days, dtype=np.int64)
    runner.OTHERS = np.asarray(reference_days, dtype=np.int64)

    params_hat_alpha_mix = runner.parse_float_tuple(
        args.params_hat_alpha_mix,
        expected_len=3,
        option_name="--params-hat-alpha-mix",
    )
    reference_sampling_config = runner.ReferenceSamplingConfig(
        mode=args.reference_sampling_mode,
        sample_size=args.reference_sample_size,
        num_trials=args.reference_num_trials,
        seed=args.reference_sampling_seed,
        history_window=args.history_window,
    )
    reference_sampling_config = runner.effective_reference_sampling_config(
        reference_sampling_config,
        reference_pool_size=len(runner.OTHERS),
    )

    args.out_root.mkdir(parents=True, exist_ok=True)
    split_payload = {
        "data_dir": str(args.data_dir),
        "user_ids": user_ids,
        "reference_days_1based": reference_days,
        "test_days_1based": test_days,
        "reference_sampling": {
            "mode": reference_sampling_config.mode,
            "sample_size": reference_sampling_config.sample_size,
            "num_trials": reference_sampling_config.num_trials,
            "seed": reference_sampling_config.seed,
            "history_window": reference_sampling_config.history_window,
        },
    }
    (args.out_root / "split_config.json").write_text(
        json.dumps(split_payload, indent=2),
        encoding="utf-8",
    )

    noise_config = runner.NoiseConfig(
        preset_name="none",
        seed=args.reference_sampling_seed,
        component_mode="none",
        strength_multiplier=0.0,
        apply_scope="all",
    )

    runner.configure_torch_runtime()
    summaries = runner.run_users(
        data_dir=args.data_dir,
        user_ids=user_ids,
        out_root=args.out_root,
        noise_config=noise_config,
        reference_sampling_config=reference_sampling_config,
        storage_dtype=runner.resolve_storage_dtype(args.storage_dtype),
        progress_interval=args.progress_interval,
        device=runner.resolve_torch_device(args.device),
        torch_dtype=runner.resolve_torch_dtype(args.torch_dtype),
        target_day_batch_size=args.target_day_batch_size,
        history_chunk_size=args.history_chunk_size,
        verify_against_cpu=args.verify_against_cpu,
        params_hat_rho=args.params_hat_rho,
        params_hat_lambda=args.params_hat_lambda,
        params_hat_alpha_mix=params_hat_alpha_mix,
    )
    aggregate = runner.aggregate_user_summaries(
        data_dir=args.data_dir,
        out_root=args.out_root,
        user_ids=user_ids,
        summaries=summaries,
    )
    (args.out_root / "all_users_summary.json").write_text(
        json.dumps(aggregate, indent=2),
        encoding="utf-8",
    )
    write_confusion_csv(args.out_root / "pooled_confusion_matrix.csv", aggregate["pooled_confusion_matrix"])
    write_table1_csv(args.out_root / "table1_source_domain_metrics.csv", aggregate)
    write_readme(args.out_root / "README.md", aggregate, args)
    print(json.dumps({
        "out_root": str(args.out_root),
        "accuracy_percent": aggregate["pooled_metrics"]["accuracy_percent"],
        "macro_f1": aggregate["pooled_metrics"]["macro_f1"],
        "balanced_accuracy": aggregate["pooled_metrics"]["balanced_accuracy"],
    }, indent=2))

if __name__ == "__main__":
    main()
