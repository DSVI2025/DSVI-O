#!/usr/bin/env python3
"""Run the fixed CUDA noise robustness suite for the best V2 parameters.

Invocation is intentionally explicit for data and output paths:

    python run_noise_sweep_suite.py --data-dir ... --output-root ...

It runs the new dataset with the current best parameters, R=1, all seven noise
families, seven nonzero strengths, one clean baseline, and sampled 50x20
reference estimation.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

DEFAULT_USER_IDS = "1,2,3,4,5,6,7,8,9,10"
DEFAULT_COMPONENTS = "additive,drift,multiplicative,cumulative,laplace,impulse,mixed"
AVAILABLE_COMPONENTS = "additive,drift,multiplicative,cumulative,laplace,impulse,mixed"
DEFAULT_STRENGTHS = "0.5,1.0,1.5,2.0,5.0,10.0,20.0"
NOISE_APPLY_SCOPES = ("all", "test-only", "reference-only")
REFERENCE_SAMPLING_MODES = ("all", "sampled")

DEFAULT_BASE_PRESET = "medium"
DEFAULT_NOISE_SEED = 20260423
DEFAULT_REFERENCE_SAMPLING_SEED = 20260423
DEFAULT_HISTORY_WINDOW = 12
DEFAULT_REFERENCE_SAMPLE_SIZE = 50
DEFAULT_REFERENCE_NUM_TRIALS = 20
DEFAULT_STORAGE_DTYPE = "float32"
DEFAULT_DEVICE = "cuda"
DEFAULT_TORCH_DTYPE = "float64"
DEFAULT_TARGET_DAY_BATCH_SIZE = 10
DEFAULT_HISTORY_CHUNK_SIZE = 1024
DEFAULT_PARAMS_HAT_RHO = 0.7
DEFAULT_PARAMS_HAT_LAMBDA = 0.305
DEFAULT_PARAMS_HAT_ALPHA_MIX = "0.75,0.58,0.16"
DEFAULT_NUM_REPEATS = 1
DEFAULT_MAX_PARALLEL = "auto"
DEFAULT_GPU_SAFETY_FRACTION = 0.85
DEFAULT_GPU_RESERVE_MB = 2048
DEFAULT_AUTO_PARALLEL_CAP = 4

RESULTS_LOCK = threading.Lock()

def parse_csv_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]

def parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]

def slugify_number(value: float | int) -> str:
    return f"{value:g}".replace("-", "neg").replace(".", "p")

def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Run the fixed CUDA V2 noise robustness suite."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing healthData.mat / insoleData.mat / EMRData.mat.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Root directory where suite outputs are written.",
    )
    parser.add_argument(
        "--user-ids",
        default=DEFAULT_USER_IDS,
        help="Comma-separated user ids to run for each experiment.",
    )
    parser.add_argument(
        "--base-preset",
        choices=["light", "medium"],
        default=DEFAULT_BASE_PRESET,
        help="Base preset that defines the reference scale for each noise family.",
    )
    parser.add_argument(
        "--noise-components",
        default=DEFAULT_COMPONENTS,
        help=f"Comma-separated component modes. Available: {AVAILABLE_COMPONENTS}.",
    )
    parser.add_argument(
        "--strengths",
        default=DEFAULT_STRENGTHS,
        help="Comma-separated nonzero strength multipliers applied on top of --base-preset.",
    )
    parser.add_argument(
        "--num-repeats",
        type=int,
        default=DEFAULT_NUM_REPEATS,
        help="Number of repeated random seeds for each noisy condition. Baseline stays single-run.",
    )
    parser.add_argument(
        "--noise-seed",
        type=int,
        default=DEFAULT_NOISE_SEED,
        help="Base seed passed into run_old_matlab_equivalent.py.",
    )
    parser.add_argument(
        "--storage-dtype",
        choices=["float32", "float64"],
        default=DEFAULT_STORAGE_DTYPE,
        help="In-memory dtype passed to the runner.",
    )
    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        help="Torch device passed to the runner.",
    )
    parser.add_argument(
        "--torch-dtype",
        choices=["float32", "float64"],
        default=DEFAULT_TORCH_DTYPE,
        help="Torch compute dtype passed to the runner.",
    )
    parser.add_argument(
        "--target-day-batch-size",
        type=int,
        default=DEFAULT_TARGET_DAY_BATCH_SIZE,
        help="Number of target days advanced together on the GPU.",
    )
    parser.add_argument(
        "--history-chunk-size",
        type=int,
        default=DEFAULT_HISTORY_CHUNK_SIZE,
        help="History time points per GPU chunk passed to the runner.",
    )
    parser.add_argument(
        "--verify-against-cpu",
        action="store_true",
        help="Ask the runner to assert day-level CPU/GPU classification parity.",
    )
    parser.add_argument(
        "--noise-apply-to",
        choices=NOISE_APPLY_SCOPES,
        default="all",
        help="Where to apply noise: all 100 days, only the 10 target test days, or only the 90 reference days.",
    )
    parser.add_argument(
        "--reference-sampling-mode",
        choices=REFERENCE_SAMPLING_MODES,
        default="sampled",
        help="Reference estimator mode passed to run_old_matlab_equivalent.py.",
    )
    parser.add_argument(
        "--reference-sample-size",
        type=int,
        default=DEFAULT_REFERENCE_SAMPLE_SIZE,
        help="Number of reference days per sampled SAA trial.",
    )
    parser.add_argument(
        "--reference-num-trials",
        type=int,
        default=DEFAULT_REFERENCE_NUM_TRIALS,
        help="Number of sampled SAA trials.",
    )
    parser.add_argument(
        "--reference-sampling-seed",
        type=int,
        default=DEFAULT_REFERENCE_SAMPLING_SEED,
        help="Seed for deterministic reference-day sampling.",
    )
    parser.add_argument(
        "--history-window",
        type=int,
        default=DEFAULT_HISTORY_WINDOW,
        help="History window L passed to run_old_matlab_equivalent.py.",
    )
    parser.add_argument(
        "--params-hat-rho",
        type=float,
        default=DEFAULT_PARAMS_HAT_RHO,
        help="Best-run params_hat.rho passed to the runner.",
    )
    parser.add_argument(
        "--params-hat-lambda",
        type=float,
        default=DEFAULT_PARAMS_HAT_LAMBDA,
        help="Best-run params_hat.lambda_ passed to the runner.",
    )
    parser.add_argument(
        "--params-hat-alpha-mix",
        default=DEFAULT_PARAMS_HAT_ALPHA_MIX,
        help="Best-run comma-separated params_hat.alpha_mix.",
    )
    parser.add_argument(
        "--max-parallel",
        default=DEFAULT_MAX_PARALLEL,
        help="Maximum concurrently running experiments. Use an integer or 'auto'.",
    )
    parser.add_argument(
        "--gpu-safety-fraction",
        type=float,
        default=DEFAULT_GPU_SAFETY_FRACTION,
        help="When --max-parallel auto is used, only this fraction of free GPU memory is considered usable.",
    )
    parser.add_argument(
        "--gpu-reserve-mb",
        type=int,
        default=DEFAULT_GPU_RESERVE_MB,
        help="Memory left unused when estimating auto parallelism.",
    )
    parser.add_argument(
        "--auto-parallel-cap",
        type=int,
        default=DEFAULT_AUTO_PARALLEL_CAP,
        help="Upper bound for auto-estimated parallel experiment count.",
    )
    parser.add_argument(
        "--skip-clean-baseline",
        action="store_true",
        help="Skip the clean baseline run.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep running remaining experiments even if one experiment fails.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write manifest and print planned experiments without running them.",
    )
    parser.set_defaults(runner_script=script_dir / "run_old_matlab_equivalent.py")
    return parser.parse_args()

def repeat_seed(base_seed: int, repeat_idx: int) -> int:
    return int(base_seed + 1000003 * repeat_idx)

def planned_experiments(args: argparse.Namespace) -> list[dict[str, Any]]:
    experiments: list[dict[str, Any]] = []
    scope_slug = args.noise_apply_to.replace("-", "")
    reference_sample_size = args.reference_sample_size
    reference_num_trials = args.reference_num_trials
    if args.reference_sampling_mode == "sampled":
        sampling_slug = f"_saa{args.reference_sample_size}x{args.reference_num_trials}"
    else:
        sampling_slug = "_allref"
        reference_sample_size = 90
        reference_num_trials = 1

    common = {
        "noise_apply_to": args.noise_apply_to,
        "reference_sampling_mode": args.reference_sampling_mode,
        "reference_sample_size": reference_sample_size,
        "reference_num_trials": reference_num_trials,
        "reference_sampling_seed": args.reference_sampling_seed,
        "history_window": args.history_window,
        "device": args.device,
        "torch_dtype": args.torch_dtype,
        "target_day_batch_size": args.target_day_batch_size,
        "history_chunk_size": args.history_chunk_size,
        "verify_against_cpu": args.verify_against_cpu,
        "params_hat_rho": args.params_hat_rho,
        "params_hat_lambda": args.params_hat_lambda,
        "params_hat_alpha_mix": args.params_hat_alpha_mix,
    }

    if not args.skip_clean_baseline:
        experiments.append(
            {
                **common,
                "name": f"baseline_clean_{scope_slug}{sampling_slug}",
                "repeat_index": 0,
                "noise_seed": args.noise_seed,
                "noise_preset": "none",
                "noise_component_mode": "none",
                "noise_strength_multiplier": 1.0,
            }
        )

    for repeat_idx in range(args.num_repeats):
        repeat_slug = "" if args.num_repeats == 1 else f"_r{repeat_idx + 1:02d}"
        for component in parse_csv_list(args.noise_components):
            for strength in parse_float_list(args.strengths):
                experiments.append(
                    {
                        **common,
                        "name": (
                            f"{component}_x{slugify_number(strength)}_"
                            f"{scope_slug}{sampling_slug}{repeat_slug}"
                        ),
                        "repeat_index": repeat_idx,
                        "noise_seed": repeat_seed(args.noise_seed, repeat_idx),
                        "noise_preset": args.base_preset,
                        "noise_component_mode": component,
                        "noise_strength_multiplier": strength,
                    }
                )
    return experiments

def assign_experiment_ordinals(experiments: list[dict[str, Any]]) -> None:
    total = len(experiments)
    for index, experiment in enumerate(experiments, start=1):
        experiment["suite_index"] = index
        experiment["suite_total"] = total

def format_experiment_progress(experiment: dict[str, Any]) -> str:
    index = int(experiment.get("suite_index", 0))
    total = int(experiment.get("suite_total", 0))
    prefix = f"[{index}/{total}]" if index and total else "[?/ ?]"
    return (
        f"{prefix} {experiment['name']} "
        f"(preset={experiment['noise_preset']}, "
        f"component={experiment['noise_component_mode']}, "
        f"strength={experiment['noise_strength_multiplier']}, "
        f"repeat={experiment['repeat_index']}, "
        f"seed={experiment['noise_seed']})"
    )

def runner_command(
    *,
    script_path: Path,
    python_exe: str,
    data_dir: Path,
    output_dir: Path,
    user_ids: str,
    storage_dtype: str,
    experiment: dict[str, Any],
) -> list[str]:
    command = [
        python_exe,
        str(script_path),
        "--data-dir",
        str(data_dir),
        "--user-ids",
        user_ids,
        "--storage-dtype",
        storage_dtype,
        "--device",
        str(experiment["device"]),
        "--torch-dtype",
        str(experiment["torch_dtype"]),
        "--noise-preset",
        str(experiment["noise_preset"]),
        "--noise-component-mode",
        str(experiment["noise_component_mode"]),
        "--noise-strength-multiplier",
        str(experiment["noise_strength_multiplier"]),
        "--noise-seed",
        str(experiment["noise_seed"]),
        "--noise-apply-to",
        str(experiment["noise_apply_to"]),
        "--reference-sampling-mode",
        str(experiment["reference_sampling_mode"]),
        "--reference-sample-size",
        str(experiment["reference_sample_size"]),
        "--reference-num-trials",
        str(experiment["reference_num_trials"]),
        "--reference-sampling-seed",
        str(experiment["reference_sampling_seed"]),
        "--history-window",
        str(experiment["history_window"]),
        "--params-hat-rho",
        str(experiment["params_hat_rho"]),
        "--params-hat-lambda",
        str(experiment["params_hat_lambda"]),
        "--params-hat-alpha-mix",
        str(experiment["params_hat_alpha_mix"]),
        "--target-day-batch-size",
        str(experiment["target_day_batch_size"]),
        "--history-chunk-size",
        str(experiment["history_chunk_size"]),
        "--out-root",
        str(output_dir),
    ]
    if bool(experiment.get("verify_against_cpu", False)):
        command.append("--verify-against-cpu")
    return command

def query_gpu_free_total_mb() -> tuple[int | None, int | None]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None, None

    first_line = completed.stdout.strip().splitlines()[0] if completed.stdout.strip() else ""
    parts = [part.strip() for part in first_line.split(",")]
    if len(parts) < 2:
        return None, None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None, None

def query_pid_gpu_memory_mb(pid: int) -> int | None:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    max_used = 0
    found = False
    for line in completed.stdout.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            row_pid = int(parts[0])
            used_mb = int(parts[1])
        except ValueError:
            continue
        if row_pid == pid:
            found = True
            max_used = max(max_used, used_mb)
    return max_used if found else 0

def run_command_with_optional_gpu_profile(
    *,
    command: list[str],
    log_path: Path,
    profile_gpu: bool,
) -> tuple[int, float, int | None]:
    started = time.time()
    peak_gpu_mb: int | None = None
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("COMMAND:\n")
        log_file.write(" ".join(command) + "\n\n")
        log_file.flush()
        process = subprocess.Popen(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if profile_gpu:
            observed_peak = 0
            while process.poll() is None:
                used_mb = query_pid_gpu_memory_mb(process.pid)
                if used_mb is not None:
                    observed_peak = max(observed_peak, used_mb)
                time.sleep(2.0)
            used_mb = query_pid_gpu_memory_mb(process.pid)
            if used_mb is not None:
                observed_peak = max(observed_peak, used_mb)
            peak_gpu_mb = observed_peak if observed_peak > 0 else None
            returncode = process.returncode
        else:
            returncode = process.wait()
    return returncode, time.time() - started, peak_gpu_mb

def run_experiment(
    *,
    script_path: Path,
    python_exe: str,
    data_dir: Path,
    output_root: Path,
    user_ids: str,
    storage_dtype: str,
    experiment: dict[str, Any],
    profile_gpu: bool = False,
) -> dict[str, Any]:
    output_dir = output_root / str(experiment["name"])
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run.log"
    command = runner_command(
        script_path=script_path,
        python_exe=python_exe,
        data_dir=data_dir,
        output_dir=output_dir,
        user_ids=user_ids,
        storage_dtype=storage_dtype,
        experiment=experiment,
    )

    returncode, duration_sec, peak_gpu_mb = run_command_with_optional_gpu_profile(
        command=command,
        log_path=log_path,
        profile_gpu=profile_gpu,
    )

    result: dict[str, Any] = {
        "name": experiment["name"],
        "suite_index": experiment.get("suite_index"),
        "suite_total": experiment.get("suite_total"),
        "repeat_index": experiment["repeat_index"],
        "noise_preset": experiment["noise_preset"],
        "noise_component_mode": experiment["noise_component_mode"],
        "noise_strength_multiplier": experiment["noise_strength_multiplier"],
        "noise_seed": experiment["noise_seed"],
        "noise_apply_to": experiment["noise_apply_to"],
        "reference_sampling_mode": experiment["reference_sampling_mode"],
        "reference_sample_size": experiment["reference_sample_size"],
        "reference_num_trials": experiment["reference_num_trials"],
        "reference_sampling_seed": experiment["reference_sampling_seed"],
        "history_window": experiment["history_window"],
        "params_hat_rho": experiment["params_hat_rho"],
        "params_hat_lambda": experiment["params_hat_lambda"],
        "params_hat_alpha_mix": experiment["params_hat_alpha_mix"],
        "device": experiment["device"],
        "torch_dtype": experiment["torch_dtype"],
        "target_day_batch_size": experiment["target_day_batch_size"],
        "history_chunk_size": experiment["history_chunk_size"],
        "verify_against_cpu": experiment["verify_against_cpu"],
        "output_dir": str(output_dir),
        "log_path": str(log_path),
        "returncode": returncode,
        "duration_sec": duration_sec,
        "peak_gpu_memory_mb": peak_gpu_mb,
        "status": "ok" if returncode == 0 else "failed",
    }

    summary_path = output_dir / "all_users_summary.json"
    if summary_path.exists():
        result["summary_path"] = str(summary_path)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        result["key_metrics"] = {
            "accuracy_percent_mean": summary["aggregate_metrics"]["accuracy_percent"]["mean"],
            "macro_f1_mean": summary["aggregate_metrics"]["macro_f1"]["mean"],
            "balanced_accuracy_mean": summary["aggregate_metrics"]["balanced_accuracy"]["mean"],
            "worst_class_recall_mean": summary["aggregate_metrics"]["worst_class_recall"]["mean"],
            "pooled_accuracy_percent": summary["pooled_metrics"]["accuracy_percent"],
            "pooled_macro_f1": summary["pooled_metrics"]["macro_f1"],
        }
    return result

def render_suite_markdown(args: argparse.Namespace, results: list[dict[str, Any]]) -> str:
    reference_sample_size = 90 if args.reference_sampling_mode == "all" else args.reference_sample_size
    reference_num_trials = 1 if args.reference_sampling_mode == "all" else args.reference_num_trials
    lines = [
        "# Noise Sweep Suite",
        "",
        f"- data_dir: `{args.data_dir}`",
        f"- output_root: `{args.output_root}`",
        f"- user_ids: `{args.user_ids}`",
        f"- base_preset: `{args.base_preset}`",
        f"- num_repeats: `{args.num_repeats}`",
        f"- noise_seed: `{args.noise_seed}`",
        f"- storage_dtype: `{args.storage_dtype}`",
        f"- device: `{args.device}`",
        f"- torch_dtype: `{args.torch_dtype}`",
        f"- target_day_batch_size: `{args.target_day_batch_size}`",
        f"- history_chunk_size: `{args.history_chunk_size}`",
        f"- verify_against_cpu: `{args.verify_against_cpu}`",
        f"- noise_apply_to: `{args.noise_apply_to}`",
        f"- reference_sampling_mode: `{args.reference_sampling_mode}`",
        f"- reference_sample_size: `{reference_sample_size}`",
        f"- reference_num_trials: `{reference_num_trials}`",
        f"- reference_sampling_seed: `{args.reference_sampling_seed}`",
        f"- history_window: `{args.history_window}`",
        f"- params_hat_rho: `{args.params_hat_rho}`",
        f"- params_hat_lambda: `{args.params_hat_lambda}`",
        f"- params_hat_alpha_mix: `{args.params_hat_alpha_mix}`",
        f"- max_parallel_requested: `{args.max_parallel}`",
        "",
        "| experiment | status | duration_sec | peak_gpu_mb | preset | component | strength | repeat | seed | ref_sampling | macro_f1_mean | balanced_acc_mean | pooled_accuracy_percent | output_dir |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for result in results:
        metrics = result.get("key_metrics", {})
        lines.append(
            f"| `{result['name']}` | `{result['status']}` | `{result['duration_sec']:.1f}` | "
            f"`{result.get('peak_gpu_memory_mb', 'NA')}` | "
            f"`{result['noise_preset']}` | `{result['noise_component_mode']}` | "
            f"`{result['noise_strength_multiplier']}` | `{result['repeat_index']}` | "
            f"`{result['noise_seed']}` | "
            f"`{result['reference_sampling_mode']}:{result['reference_sample_size']}x{result['reference_num_trials']}` | "
            f"`{metrics.get('macro_f1_mean', 'NA')}` | "
            f"`{metrics.get('balanced_accuracy_mean', 'NA')}` | "
            f"`{metrics.get('pooled_accuracy_percent', 'NA')}` | "
            f"`{result['output_dir']}` |"
        )
    lines.append("")
    return "\n".join(lines)

def write_suite_outputs(
    *,
    output_root: Path,
    args: argparse.Namespace,
    results: list[dict[str, Any]],
) -> None:
    with RESULTS_LOCK:
        sorted_results = sorted(results, key=lambda item: item["name"])
        (output_root / "suite_results.json").write_text(
            json.dumps({"results": sorted_results}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (output_root / "suite_report.md").write_text(
            render_suite_markdown(args, sorted_results),
            encoding="utf-8",
        )

def estimate_parallelism_from_profile(
    *,
    requested: str,
    peak_gpu_mb: int | None,
    safety_fraction: float,
    reserve_mb: int,
    cap: int,
) -> tuple[int, dict[str, Any]]:
    if requested != "auto":
        try:
            value = int(requested)
        except ValueError as exc:
            raise ValueError("--max-parallel must be 'auto' or a positive integer.") from exc
        if value <= 0:
            raise ValueError("--max-parallel must be positive.")
        return value, {"mode": "manual", "requested": requested, "actual": value}

    free_mb, total_mb = query_gpu_free_total_mb()
    profile = {
        "mode": "auto",
        "requested": requested,
        "gpu_free_mb_after_profile": free_mb,
        "gpu_total_mb": total_mb,
        "profile_peak_gpu_memory_mb": peak_gpu_mb,
        "gpu_safety_fraction": safety_fraction,
        "gpu_reserve_mb": reserve_mb,
        "auto_parallel_cap": cap,
        "actual": 1,
        "reason": "",
    }
    if peak_gpu_mb is None or peak_gpu_mb <= 0:
        profile["reason"] = "profile_peak_gpu_memory_mb unavailable; falling back to serial."
        return 1, profile
    if free_mb is None:
        profile["reason"] = "nvidia-smi free memory unavailable; falling back to serial."
        return 1, profile

    usable_mb = max(0.0, float(free_mb - reserve_mb) * float(safety_fraction))
    estimated = int(math.floor(usable_mb / float(peak_gpu_mb)))
    actual = max(1, min(int(cap), estimated))
    profile["usable_gpu_memory_mb"] = usable_mb
    profile["estimated_parallel"] = estimated
    profile["actual"] = actual
    profile["reason"] = "auto-estimated from profiled baseline run."
    return actual, profile

def write_manifest(
    *,
    output_root: Path,
    args: argparse.Namespace,
    experiments: list[dict[str, Any]],
    parallel_profile: dict[str, Any],
) -> None:
    reference_sample_size = 90 if args.reference_sampling_mode == "all" else args.reference_sample_size
    reference_num_trials = 1 if args.reference_sampling_mode == "all" else args.reference_num_trials
    suite_manifest = {
        "data_dir": str(args.data_dir),
        "output_root": str(output_root),
        "user_ids": parse_csv_list(args.user_ids),
        "base_preset": args.base_preset,
        "noise_components": parse_csv_list(args.noise_components),
        "strengths": parse_float_list(args.strengths),
        "num_repeats": args.num_repeats,
        "noise_seed": args.noise_seed,
        "storage_dtype": args.storage_dtype,
        "device": args.device,
        "torch_dtype": args.torch_dtype,
        "target_day_batch_size": args.target_day_batch_size,
        "history_chunk_size": args.history_chunk_size,
        "verify_against_cpu": args.verify_against_cpu,
        "noise_apply_to": args.noise_apply_to,
        "reference_sampling_mode": args.reference_sampling_mode,
        "reference_sample_size": reference_sample_size,
        "reference_num_trials": reference_num_trials,
        "reference_sampling_seed": args.reference_sampling_seed,
        "history_window": args.history_window,
        "params_hat": {
            "rho": args.params_hat_rho,
            "lambda": args.params_hat_lambda,
            "alpha_mix": args.params_hat_alpha_mix,
        },
        "parallel": parallel_profile,
        "runner_script": str(args.runner_script),
        "python_executable": sys.executable,
        "experiments": experiments,
    }
    (output_root / "suite_manifest.json").write_text(
        json.dumps(suite_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

def run_experiments_parallel(
    *,
    args: argparse.Namespace,
    output_root: Path,
    experiments: list[dict[str, Any]],
    max_parallel: int,
    initial_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    results = list(initial_results)
    if not experiments:
        return results

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel) as executor:
        experiment_iter = iter(experiments)
        future_to_experiment: dict[concurrent.futures.Future, dict[str, Any]] = {}

        def submit_next() -> bool:
            try:
                experiment = next(experiment_iter)
            except StopIteration:
                return False
            print(
                f"[suite] START {format_experiment_progress(experiment)} "
                f"-> {output_root / str(experiment['name'])}",
                flush=True,
            )
            future = executor.submit(
                run_experiment,
                script_path=args.runner_script,
                python_exe=sys.executable,
                data_dir=args.data_dir,
                output_root=output_root,
                user_ids=args.user_ids,
                storage_dtype=args.storage_dtype,
                experiment=experiment,
                profile_gpu=False,
            )
            future_to_experiment[future] = experiment
            return True

        for _ in range(max_parallel):
            if not submit_next():
                break

        while future_to_experiment:
            done, _pending = concurrent.futures.wait(
                future_to_experiment,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for future in done:
                experiment = future_to_experiment.pop(future)
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001 - record hard worker failure in suite outputs.
                    result = {
                        "name": experiment["name"],
                        "suite_index": experiment.get("suite_index"),
                        "suite_total": experiment.get("suite_total"),
                        "repeat_index": experiment["repeat_index"],
                        "noise_preset": experiment["noise_preset"],
                        "noise_component_mode": experiment["noise_component_mode"],
                        "noise_strength_multiplier": experiment["noise_strength_multiplier"],
                        "noise_seed": experiment["noise_seed"],
                        "status": "failed",
                        "returncode": None,
                        "duration_sec": 0.0,
                        "error": repr(exc),
                        "output_dir": str(output_root / str(experiment["name"])),
                    }
                results.append(result)
                write_suite_outputs(output_root=output_root, args=args, results=results)
                metrics = result.get("key_metrics", {})
                metric_text = ""
                if metrics:
                    metric_text = (
                        f" pooled_acc={metrics.get('pooled_accuracy_percent', 'NA')}"
                        f" pooled_macro_f1={metrics.get('pooled_macro_f1', 'NA')}"
                    )
                print(
                    f"[suite] DONE  {format_experiment_progress(experiment)} "
                    f"status={result['status']} duration={result['duration_sec']:.1f}s"
                    f"{metric_text}",
                    flush=True,
                )
                if result["status"] != "ok" and not args.continue_on_error:
                    raise SystemExit(
                        f"Experiment {result['name']} failed. "
                        f"See log: {result.get('log_path', 'no log path')}"
                    )
                submit_next()
    return results

def main() -> None:
    args = parse_args()
    if args.num_repeats <= 0:
        raise ValueError("--num-repeats must be positive.")
    if args.target_day_batch_size <= 0:
        raise ValueError("--target-day-batch-size must be positive.")
    if args.history_chunk_size <= 0:
        raise ValueError("--history-chunk-size must be positive.")

    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    experiments = planned_experiments(args)
    assign_experiment_ordinals(experiments)
    print(
        f"[suite] planned {len(experiments)} experiments | "
        f"data_dir={args.data_dir} | output_root={output_root} | "
        f"users={args.user_ids} | noise_apply_to={args.noise_apply_to} | "
        f"R={args.num_repeats} | max_parallel={args.max_parallel}",
        flush=True,
    )

    if args.dry_run:
        dry_profile = {
            "mode": args.max_parallel,
            "requested": args.max_parallel,
            "actual": 0,
            "reason": "dry-run only",
        }
        write_manifest(
            output_root=output_root,
            args=args,
            experiments=experiments,
            parallel_profile=dry_profile,
        )
        print(json.dumps({"experiments": experiments}, ensure_ascii=False, indent=2))
        print(f"Dry-run manifest: {output_root / 'suite_manifest.json'}")
        return

    initial_results: list[dict[str, Any]] = []
    remaining_experiments = list(experiments)
    peak_gpu_mb: int | None = None

    if args.max_parallel == "auto" and remaining_experiments:
        profile_experiment = remaining_experiments.pop(0)
        print(
            f"[suite] PROFILE START {format_experiment_progress(profile_experiment)} "
            "for auto parallelism",
            flush=True,
        )
        profile_result = run_experiment(
            script_path=args.runner_script,
            python_exe=sys.executable,
            data_dir=args.data_dir,
            output_root=output_root,
            user_ids=args.user_ids,
            storage_dtype=args.storage_dtype,
            experiment=profile_experiment,
            profile_gpu=True,
        )
        initial_results.append(profile_result)
        peak_gpu_mb = profile_result.get("peak_gpu_memory_mb")
        write_suite_outputs(output_root=output_root, args=args, results=initial_results)
        metrics = profile_result.get("key_metrics", {})
        metric_text = ""
        if metrics:
            metric_text = (
                f" pooled_acc={metrics.get('pooled_accuracy_percent', 'NA')}"
                f" pooled_macro_f1={metrics.get('pooled_macro_f1', 'NA')}"
            )
        print(
            f"[suite] PROFILE DONE  {format_experiment_progress(profile_experiment)} "
            f"status={profile_result['status']} duration={profile_result['duration_sec']:.1f}s "
            f"peak_gpu_mb={peak_gpu_mb}{metric_text}",
            flush=True,
        )
        if profile_result["status"] != "ok" and not args.continue_on_error:
            raise SystemExit(
                f"Experiment {profile_result['name']} failed. "
                f"See log: {profile_result.get('log_path', 'no log path')}"
            )

    max_parallel, parallel_profile = estimate_parallelism_from_profile(
        requested=args.max_parallel,
        peak_gpu_mb=peak_gpu_mb,
        safety_fraction=args.gpu_safety_fraction,
        reserve_mb=args.gpu_reserve_mb,
        cap=args.auto_parallel_cap,
    )
    write_manifest(
        output_root=output_root,
        args=args,
        experiments=experiments,
        parallel_profile=parallel_profile,
    )

    print(
        f"[suite] running {len(remaining_experiments)} remaining experiments "
        f"with max_parallel={max_parallel} | parallel_profile={parallel_profile}",
        flush=True,
    )
    results = run_experiments_parallel(
        args=args,
        output_root=output_root,
        experiments=remaining_experiments,
        max_parallel=max_parallel,
        initial_results=initial_results,
    )
    write_suite_outputs(output_root=output_root, args=args, results=results)

    print(f"Suite manifest : {output_root / 'suite_manifest.json'}")
    print(f"Suite results  : {output_root / 'suite_results.json'}")
    print(f"Suite report   : {output_root / 'suite_report.md'}")
    print("Suite finished.")

if __name__ == "__main__":
    main()
