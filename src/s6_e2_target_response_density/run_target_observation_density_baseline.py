from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

torch = None

DEFAULT_USER_IDS = "11,12,13,14,15,16,17,18,19,20"
DEFAULT_STRIDES = "120,240,720,1440"

def parse_int_list(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one integer.")
    return values

def parse_float_tuple(raw: str, *, expected_len: int, option_name: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in raw.split(",") if item.strip())
    if len(values) != expected_len:
        raise ValueError(
            f"{option_name} must contain exactly {expected_len} comma-separated values; "
            f"received {len(values)}."
        )
    return values

def load_runner(code_root: Path):
    runner_path = code_root / "run_old_matlab_equivalent.py"
    if not runner_path.exists():
        raise FileNotFoundError(f"Missing runner: {runner_path}")
    module_name = "_target_density_runner_old_matlab_equivalent"
    spec = importlib.util.spec_from_file_location(module_name, runner_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load runner module from {runner_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def require_torch():
    global torch
    if torch is None:
        import torch as torch_module

        torch = torch_module
    return torch

def class_accuracy_from_confusion(confusion: np.ndarray) -> list[float]:
    total = float(confusion.sum())
    values: list[float] = []
    for idx in range(3):
        tp = float(confusion[idx, idx])
        fp = float(confusion[:, idx].sum() - confusion[idx, idx])
        fn = float(confusion[idx, :].sum() - confusion[idx, idx])
        tn = total - tp - fp - fn
        values.append((tp + tn) / max(total, 1.0))
    return values

def density_label(stride_steps: int) -> str:
    if stride_steps == 1:
        return "Full target-only (5 sec)"
    minutes = stride_steps * 5.0 / 60.0
    if minutes >= 60.0 and float(minutes / 60.0).is_integer():
        hours = int(minutes / 60.0)
        unit = "h"
        return f"Sparse target-only ({hours} {unit})"
    if float(minutes).is_integer():
        return f"Sparse target-only ({int(minutes)} min)"
    return f"Sparse target-only ({minutes:g} min)"

def interval_minutes(stride_steps: int) -> float:
    return float(stride_steps) * 5.0 / 60.0

def compute_sparse_day_batch(
    runner,
    *,
    day_indices_1based: list[int],
    user_id: int,
    h_all: list[torch.Tensor],
    labels_all: torch.Tensor,
    reference_trial_idx: torch.Tensor,
    history_precomputes: list[Any],
    params_true: Any,
    params_hat: Any,
    t_day: int,
    response_stride_steps: int,
    x_bounds: tuple[float, float],
    progress_interval: int,
    history_chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    day_idx = torch.as_tensor(
        [int(day) - 1 for day in day_indices_1based],
        device=labels_all.device,
        dtype=torch.long,
    )
    dtype = labels_all.dtype
    device = labels_all.device
    num_days = int(day_idx.numel())
    b_days = labels_all[day_idx, :]
    x = torch.zeros((num_days, t_day + 1), device=device, dtype=dtype)
    xhat = torch.zeros((num_days, t_day + 1), device=device, dtype=dtype)
    xhat[:, 0] = 0.5
    lb, ub = x_bounds

    theta_true = torch.tensor(params_true.theta, device=device, dtype=dtype)
    b3 = h_all[2][day_idx, :]
    hsamp3 = h_all[2][reference_trial_idx]
    csamp3 = torch.mean(labels_all[reference_trial_idx], dim=2)
    h3_th, h3_tc, h3_tn = runner.precompute_static_modal_terms_days_torch(
        b=b3,
        hsamp=hsamp3,
        csamp=csamp3,
        params_hat=params_hat,
    )

    last_y1_trials = None
    last_y2_trials = None
    last_y3_trials = None
    response_updates = 0
    started = time.time()

    for t in range(t_day):
        m_t = b_days[:, t]
        s_true = 0.5 * (1.0 + torch.tanh((m_t[:, None] - theta_true[None, :]) / params_true.sigma))
        ell_true = torch.sum(s_true, dim=1)
        q_true = (
            params_true.A[0] * np.sin(2.0 * np.pi * params_true.f[0] * t)
            + params_true.A[1] * np.sin(2.0 * np.pi * params_true.f[1] * t)
        )
        x[:, t + 1] = torch.clamp(
            params_true.alpha_ou * x[:, t] + (1.0 - params_true.alpha_ou) * (ell_true + q_true),
            min=lb,
            max=ub,
        )

        b1 = h_all[0][day_idx, :, t]
        b2 = h_all[1][day_idx, :, t]
        if last_y1_trials is None or t % response_stride_steps == 0:
            last_y1_trials = runner.solve_history_modal_y_days_torch(
                b=b1,
                xhat_t=xhat[:, t],
                cache=history_precomputes[0],
                current_t=t,
                params_hat=params_hat,
                history_chunk_size=history_chunk_size,
            )
            last_y2_trials = runner.solve_history_modal_y_days_torch(
                b=b2,
                xhat_t=xhat[:, t],
                cache=history_precomputes[1],
                current_t=t,
                params_hat=params_hat,
                history_chunk_size=history_chunk_size,
            )
            last_y3_trials = runner.solve_static_modal_y_days_torch(
                b=b3,
                h_th=h3_th,
                h_tc=h3_tc,
                h_tn=h3_tn,
                xhat_t=xhat[:, t],
                params_hat=params_hat,
            )
            response_updates += 1

        mhat_trials = (
            params_hat.alpha_mix[0] * torch.sum(last_y1_trials * b1[:, None, :], dim=2)
            + params_hat.alpha_mix[1] * torch.sum(last_y2_trials * b2[:, None, :], dim=2)
            + params_hat.alpha_mix[2] * torch.sum(last_y3_trials * b3[:, None, :], dim=2)
        )
        ell_hat = torch.mean(mhat_trials, dim=1)
        q_hat = (
            params_hat.A[0] * np.sin(2.0 * np.pi * params_hat.f[0] * t)
            + params_hat.A[1] * np.sin(2.0 * np.pi * params_hat.f[1] * t)
        )
        xhat[:, t + 1] = torch.clamp(
            params_hat.alpha_ou * xhat[:, t] + (1.0 - params_hat.alpha_ou) * (ell_hat + q_hat),
            min=lb,
            max=ub,
        )
        if progress_interval > 0 and (
            t == 0 or (t + 1) % progress_interval == 0 or (t + 1) == t_day
        ):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.time() - started
            pct = 100.0 * float(t + 1) / float(t_day)
            print(
                f"[user {user_id}] stride={response_stride_steps} day_batch={day_indices_1based} "
                f"t={t + 1}/{t_day} ({pct:.1f}%) "
                f"updates={response_updates} "
                f"xhat_mean={float(torch.mean(xhat[:, t + 1]).detach().cpu()):.6f} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    runtime_seconds = time.time() - started
    x_np = x[:, 1:].detach().cpu().numpy()
    xhat_np = xhat[:, 1:].detach().cpu().numpy()
    x_sm = np.stack([runner.smooth_gaussian_same(row, window=21) for row in x_np], axis=0)
    xhat_sm = np.stack([runner.smooth_gaussian_same(row, window=21) for row in xhat_np], axis=0)
    return x_sm, xhat_sm, runtime_seconds, response_updates

def run_sparse_user(
    runner,
    *,
    data_dir: Path,
    user_id: int,
    out_dir: Path,
    response_stride_steps: int,
    storage_dtype: np.dtype,
    progress_interval: int,
    device: torch.device,
    torch_dtype: torch.dtype,
    target_day_batch_size: int,
    history_chunk_size: int,
    params_hat_rho: float,
    params_hat_lambda: float,
    params_hat_alpha_mix: tuple[float, float, float],
    reference_sampling_config: Any,
    time_step_limit: int | None,
) -> dict[str, Any]:
    user_started = time.time()
    params_true = runner.ParamsTrue(
        alpha_ou=float(np.exp(-4.0)),
        A=(0.15, 0.17),
        f=(0.025, 0.021),
        sigma=0.5,
        theta=(0.5, 1.5),
    )
    params_hat = runner.ParamsHat(
        alpha_ou=float(np.exp(-4.0)),
        A=(0.11, 0.13),
        f=(0.035, 0.038),
        sigma=0.4,
        theta=(0.5, 1.5),
        rho=float(params_hat_rho),
        lambda_=float(params_hat_lambda),
        alpha_mix=tuple(float(value) for value in params_hat_alpha_mix),
    )
    h_all, labels_all = runner.load_user_tensors(data_dir, user_id, storage_dtype)
    t_day_full = int(labels_all.shape[1])
    t_day = t_day_full if time_step_limit is None else min(t_day_full, int(time_step_limit))
    if t_day <= 0:
        raise ValueError("--time-step-limit must be positive when provided.")

    x_all = np.zeros((len(runner.DAYS), t_day), dtype=np.float64)
    xhat_all = np.zeros((len(runner.DAYS), t_day), dtype=np.float64)
    sample_idx = runner.OTHERS.astype(np.int64) - 1
    reference_trial_indices = runner.build_reference_trial_indices(
        sample_idx=sample_idx,
        config=reference_sampling_config,
        user_id=user_id,
    )
    h_all_torch = [torch.as_tensor(arr, device=device, dtype=torch_dtype) for arr in h_all]
    labels_all_torch = torch.as_tensor(labels_all, device=device, dtype=torch_dtype)
    reference_trial_idx_torch = torch.as_tensor(
        np.stack(reference_trial_indices, axis=0),
        device=device,
        dtype=torch.long,
    )
    history_precomputes = [
        runner.build_torch_history_modal_precompute(
            h_source=h_all_torch[i],
            labels_all=labels_all_torch,
            trial_idx=reference_trial_idx_torch,
            history_window=reference_sampling_config.history_window,
        )
        for i in range(2)
    ]

    out_dir.mkdir(parents=True, exist_ok=True)
    day_out_dir = out_dir / "day_results"
    day_out_dir.mkdir(parents=True, exist_ok=True)
    day_summaries: list[dict[str, Any]] = []
    days_list = runner.DAYS.tolist()
    batch_size = max(1, int(target_day_batch_size))
    target_days_started = time.time()
    total_response_updates = 0

    for batch_start in range(0, len(days_list), batch_size):
        batch_days = days_list[batch_start : batch_start + batch_size]
        print(
            f"[user {user_id}] stride={response_stride_steps} day batch "
            f"{batch_start + 1}-{batch_start + len(batch_days)}/{len(days_list)} "
            f"-> day_idx={batch_days}",
            flush=True,
        )
        x_batch, xhat_batch, batch_runtime_seconds, response_updates = compute_sparse_day_batch(
            runner,
            day_indices_1based=batch_days,
            user_id=user_id,
            h_all=h_all_torch,
            labels_all=labels_all_torch,
            reference_trial_idx=reference_trial_idx_torch,
            history_precomputes=history_precomputes,
            params_true=params_true,
            params_hat=params_hat,
            t_day=t_day,
            response_stride_steps=response_stride_steps,
            x_bounds=(0.0, 2.0),
            progress_interval=progress_interval,
            history_chunk_size=history_chunk_size,
        )
        total_response_updates += response_updates * len(batch_days)
        print(
            f"[user {user_id}] stride={response_stride_steps} day batch "
            f"{batch_start + 1}-{batch_start + len(batch_days)}/{len(days_list)} "
            f"runtime={batch_runtime_seconds:.3f}s updates={response_updates}",
            flush=True,
        )
        for batch_offset, day in enumerate(batch_days):
            idx = batch_start + batch_offset
            x_sm = x_batch[batch_offset]
            xhat_sm = xhat_batch[batch_offset]
            x_all[idx, :] = x_sm
            xhat_all[idx, :] = xhat_sm
            day_true_disc = runner.discretize_three_level(x_sm.reshape(1, -1))
            day_pred_disc = runner.discretize_three_level(xhat_sm.reshape(1, -1))
            day_c = runner.confusion_matrix_3(day_true_disc, day_pred_disc)
            day_metrics = runner.classification_metrics_from_confusion_matrix(day_c)
            day_stem = f"day{idx + 1:02d}_idx{int(day):03d}"
            day_npz_path = day_out_dir / f"{day_stem}_outputs.npz"
            np.savez_compressed(
                day_npz_path,
                x=x_sm,
                xhat=xhat_sm,
                true_disc=day_true_disc[0],
                pred_disc=day_pred_disc[0],
                day_1based=int(day),
                response_stride_steps=response_stride_steps,
            )
            day_summary = {
                "user_id": user_id,
                "day_order": idx + 1,
                "num_days": len(runner.DAYS),
                "day_1based": int(day),
                "t_day": t_day,
                "t_day_full": t_day_full,
                "response_stride_steps": response_stride_steps,
                "response_update_interval_minutes": interval_minutes(response_stride_steps),
                "accuracy_percent": day_metrics["accuracy"] * 100.0,
                "confusion_matrix": day_c.tolist(),
                "precision": day_metrics["precision"],
                "recall": day_metrics["recall"],
                "specificity": day_metrics["specificity"],
                "f1": day_metrics["f1"],
                "support": day_metrics["support"],
                "macro_precision": day_metrics["macro_precision"],
                "macro_recall": day_metrics["macro_recall"],
                "macro_f1": day_metrics["macro_f1"],
                "balanced_accuracy": day_metrics["balanced_accuracy"],
                "weighted_f1": day_metrics["weighted_f1"],
                "worst_class_recall": day_metrics["worst_class_recall"],
                "worst_class_f1": day_metrics["worst_class_f1"],
                "x_min": float(np.min(x_sm)),
                "x_max": float(np.max(x_sm)),
                "x_final": float(x_sm[-1]),
                "xhat_min": float(np.min(xhat_sm)),
                "xhat_max": float(np.max(xhat_sm)),
                "xhat_final": float(xhat_sm[-1]),
                "runtime_seconds": batch_runtime_seconds / float(len(batch_days)),
                "runtime_minutes": batch_runtime_seconds / float(len(batch_days)) / 60.0,
                "runtime_note": "amortized over target day batch",
                "target_day_batch_size": len(batch_days),
                "target_day_batch_runtime_seconds": batch_runtime_seconds,
                "outputs_npz": str(day_npz_path),
            }
            (day_out_dir / f"{day_stem}_summary.json").write_text(
                json.dumps(day_summary, indent=2),
                encoding="utf-8",
            )
            runner.write_confusion_matrix_csv(day_out_dir / f"{day_stem}_confusion_matrix.csv", day_c)
            day_summaries.append(day_summary)
            print(
                f"[user {user_id}] stride={response_stride_steps} "
                f"day {idx + 1}/{len(runner.DAYS)} result -> "
                f"accuracy={day_summary['accuracy_percent']:.4f}% "
                f"macro_f1={day_summary['macro_f1']:.6f} "
                f"balanced_acc={day_summary['balanced_accuracy']:.6f}",
                flush=True,
            )

    target_days_runtime_seconds = time.time() - target_days_started
    true_disc = runner.discretize_three_level(x_all)
    pred_disc = runner.discretize_three_level(xhat_all)
    c = runner.confusion_matrix_3(true_disc, pred_disc)
    metrics = runner.classification_metrics_from_confusion_matrix(c)
    np.savez_compressed(
        out_dir / "user_run_outputs.npz",
        x_all=x_all,
        xhat_all=xhat_all,
        true_disc=true_disc,
        pred_disc=pred_disc,
        days=runner.DAYS,
        response_stride_steps=response_stride_steps,
    )
    runtime_seconds = time.time() - user_started
    summary = {
        "user_id": user_id,
        "days_1based": runner.DAYS.tolist(),
        "sample_days_1based": runner.OTHERS.tolist(),
        "reference_sampling": {
            "mode": reference_sampling_config.mode,
            "sample_size": reference_sampling_config.sample_size,
            "num_trials": reference_sampling_config.num_trials,
            "seed": reference_sampling_config.seed,
            "history_window": reference_sampling_config.history_window,
        },
        "t_day": t_day,
        "t_day_full": t_day_full,
        "runtime_seconds": runtime_seconds,
        "runtime_minutes": runtime_seconds / 60.0,
        "target_days_runtime_seconds": target_days_runtime_seconds,
        "target_days_runtime_minutes": target_days_runtime_seconds / 60.0,
        "accuracy_percent": metrics["accuracy"] * 100.0,
        "confusion_matrix": c.tolist(),
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "specificity": metrics["specificity"],
        "f1": metrics["f1"],
        "support": metrics["support"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "weighted_f1": metrics["weighted_f1"],
        "worst_class_recall": metrics["worst_class_recall"],
        "worst_class_f1": metrics["worst_class_f1"],
        "day_summaries": day_summaries,
        "noise": {
            "preset": "none",
            "seed": reference_sampling_config.seed,
            "component_mode": "none",
            "strength_multiplier": 0.0,
            "apply_scope": "all",
        },
        "observation_density": {
            "response_stride_steps": response_stride_steps,
            "response_update_interval_minutes": interval_minutes(response_stride_steps),
            "evaluation_grid": "five_second_full_day",
            "carry_forward": "target_response_weights",
            "total_response_updates": total_response_updates,
        },
        "compute": {
            "backend": "torch_sparse_target_response",
            "device": str(device),
            "torch_dtype": str(torch_dtype).replace("torch.", ""),
            "target_day_batch_size": target_day_batch_size,
            "history_chunk_size": history_chunk_size,
            "time_step_limit": time_step_limit,
        },
        "model": {
            "params_hat": {
                "rho": params_hat.rho,
                "lambda_": params_hat.lambda_,
                "alpha_mix": list(params_hat.alpha_mix),
            }
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    runner.write_confusion_matrix_csv(out_dir / "confusion_matrix.csv", c)
    runner.write_user_markdown_report(out_dir / "report.md", summary)
    print(
        f"[user {user_id}] stride={response_stride_steps} runtime={runtime_seconds:.3f}s "
        f"({runtime_seconds / 60.0:.3f} min) accuracy={summary['accuracy_percent']:.6f}%",
        flush=True,
    )
    return summary

def load_aggregate(path: Path) -> dict[str, Any]:
    return json.loads((path / "all_users_summary.json").read_text(encoding="utf-8"))

def aggregate_runtime_seconds(aggregate: dict[str, Any]) -> float:
    summaries = aggregate.get("summaries", [])
    if summaries:
        return float(np.mean([float(item["target_days_runtime_seconds"]) for item in summaries]))
    runtime = aggregate.get("runtime", {})
    return float(runtime.get("mean_seconds", 0.0))

def method_summary_row(
    *,
    label: str,
    stride_steps: int,
    aggregate: dict[str, Any],
    result_root: Path,
) -> dict[str, Any]:
    pooled = aggregate["pooled_metrics"]
    return {
        "density_label": label,
        "response_stride_steps": stride_steps,
        "response_update_interval_minutes": interval_minutes(stride_steps),
        "accuracy_percent": pooled["accuracy_percent"],
        "macro_f1": pooled["macro_f1"],
        "balanced_accuracy": pooled["balanced_accuracy"],
        "weighted_f1": pooled["weighted_f1"],
        "worst_class_recall": pooled["worst_class_recall"],
        "worst_class_f1": pooled["worst_class_f1"],
        "target_days_runtime_seconds_mean_per_user": aggregate_runtime_seconds(aggregate),
        "result_root": str(result_root),
    }

def class_metric_rows(
    *,
    label: str,
    stride_steps: int,
    aggregate: dict[str, Any],
) -> list[dict[str, Any]]:
    pooled = aggregate["pooled_metrics"]
    confusion = np.asarray(aggregate["pooled_confusion_matrix"], dtype=np.int64)
    class_acc = class_accuracy_from_confusion(confusion)
    class_names = ["healthy (0)", "weak (1)", "ill (2)"]
    rows: list[dict[str, Any]] = []
    for idx, class_name in enumerate(class_names):
        rows.append(
            {
                "density_label": label,
                "response_stride_steps": stride_steps,
                "response_update_interval_minutes": interval_minutes(stride_steps),
                "class": class_name,
                "class_index": idx,
                "accuracy": class_acc[idx],
                "precision": pooled["precision"][idx],
                "recall": pooled["recall"][idx],
                "specificity": pooled["specificity"][idx],
                "f1": pooled["f1"][idx],
                "support": pooled["support"][idx],
            }
        )
    return rows

def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def write_latex_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    lines: list[str] = []
    for row in rows:
        label = str(row["density_label"])
        class_name = str(row["class"])
        if class_name.startswith("healthy"):
            first_col = f"\\multirow{{3}}{{*}}{{{label}}}"
        else:
            first_col = ""
        lines.append(
            f"{first_col} & {class_name} & "
            f"{float(row['accuracy']):.4f} & "
            f"{float(row['precision']):.4f} & "
            f"{float(row['recall']):.4f} & "
            f"{float(row['specificity']):.4f} & "
            f"{float(row['f1']):.4f} \\\\"
        )
        if class_name.startswith("ill"):
            lines.append("\\midrule")
    if lines and lines[-1] == "\\midrule":
        lines.pop()
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def write_summary_markdown(path: Path, method_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# S6-E2a Target Response-Density Baseline",
        "",
        "Target-specific second-stage responses are recomputed only on the listed",
        "target update grid and are carried forward between update times. Metrics",
        "are evaluated on the original five-second target-day grid.",
        "",
        "| baseline | stride steps | interval min | accuracy (%) | macro-F1 | balanced acc. | mean target runtime s/user |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in method_rows:
        lines.append(
            f"| {row['density_label']} | {int(row['response_stride_steps'])} | "
            f"{float(row['response_update_interval_minutes']):.6g} | "
            f"{float(row['accuracy_percent']):.6f} | "
            f"{float(row['macro_f1']):.6f} | "
            f"{float(row['balanced_accuracy']):.6f} | "
            f"{float(row['target_days_runtime_seconds_mean_per_user']):.6f} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")

def main() -> None:
    stable_dir = Path(__file__).resolve().parents[2]
    default_code_root = stable_dir / "src" / "s6_e2_target_full_recomputation"
    default_data_dir = stable_dir / "data" / "mat" / "source_target_shared_emr"
    default_full_baseline_root = stable_dir / "results" / "s6_e2_target_full_recomputation"
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-root", type=Path, default=default_code_root)
    parser.add_argument("--data-dir", type=Path, default=default_data_dir)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--full-baseline-root", type=Path, default=default_full_baseline_root)
    parser.add_argument("--user-ids", default=DEFAULT_USER_IDS)
    parser.add_argument("--response-strides", default=DEFAULT_STRIDES)
    parser.add_argument("--storage-dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--target-day-batch-size", type=int, default=10)
    parser.add_argument("--history-chunk-size", type=int, default=1024)
    parser.add_argument("--progress-interval", type=int, default=0)
    parser.add_argument("--reference-sample-size", type=int, default=50)
    parser.add_argument("--reference-num-trials", type=int, default=20)
    parser.add_argument("--reference-sampling-seed", type=int, default=20260423)
    parser.add_argument("--history-window", type=int, default=12)
    parser.add_argument("--params-hat-rho", type=float, default=0.7)
    parser.add_argument("--params-hat-lambda", type=float, default=0.305)
    parser.add_argument("--params-hat-alpha-mix", default="0.75,0.58,0.16")
    parser.add_argument("--time-step-limit", type=int, default=None)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse a stride directory when its all_users_summary.json already exists.",
    )
    args = parser.parse_args()

    if args.target_day_batch_size <= 0:
        raise ValueError("--target-day-batch-size must be positive.")
    if args.history_chunk_size <= 0:
        raise ValueError("--history-chunk-size must be positive.")
    user_ids = parse_int_list(args.user_ids)
    strides = parse_int_list(args.response_strides)
    if any(stride <= 0 for stride in strides):
        raise ValueError("--response-strides must contain positive integers.")
    params_hat_alpha_mix = parse_float_tuple(
        args.params_hat_alpha_mix,
        expected_len=3,
        option_name="--params-hat-alpha-mix",
    )

    runner = load_runner(args.code_root)
    require_torch()
    runner.configure_torch_runtime()
    storage_dtype = runner.resolve_storage_dtype(args.storage_dtype)
    torch_dtype = runner.resolve_torch_dtype(args.torch_dtype)
    device = runner.resolve_torch_device(args.device)
    reference_sampling_config = runner.ReferenceSamplingConfig(
        mode="sampled",
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

    print(f"Code root: {args.code_root}", flush=True)
    print(f"Data dir: {args.data_dir}", flush=True)
    print(f"Output root: {args.out_root}", flush=True)
    print(f"Users: {user_ids}", flush=True)
    print(f"Response strides: {strides}", flush=True)
    print(f"Device: {device}; torch_dtype={args.torch_dtype}; storage_dtype={args.storage_dtype}", flush=True)

    aggregates_by_stride: dict[int, dict[str, Any]] = {}
    result_roots_by_stride: dict[int, Path] = {}
    for stride in strides:
        stride_root = args.out_root / f"stride_{stride:03d}"
        result_roots_by_stride[stride] = stride_root
        summary_path = stride_root / "all_users_summary.json"
        if args.skip_existing and summary_path.exists():
            print(f"Reusing existing stride={stride} result at {stride_root}", flush=True)
            aggregates_by_stride[stride] = load_aggregate(stride_root)
            continue

        summaries: list[dict[str, Any]] = []
        for user_id in user_ids:
            user_out_dir = stride_root / f"user{user_id:02d}"
            print(f"=== Running stride={stride} user {user_id} -> {user_out_dir} ===", flush=True)
            summary = run_sparse_user(
                runner,
                data_dir=args.data_dir,
                user_id=user_id,
                out_dir=user_out_dir,
                response_stride_steps=stride,
                storage_dtype=storage_dtype,
                progress_interval=args.progress_interval,
                device=device,
                torch_dtype=torch_dtype,
                target_day_batch_size=args.target_day_batch_size,
                history_chunk_size=args.history_chunk_size,
                params_hat_rho=args.params_hat_rho,
                params_hat_lambda=args.params_hat_lambda,
                params_hat_alpha_mix=params_hat_alpha_mix,
                reference_sampling_config=reference_sampling_config,
                time_step_limit=args.time_step_limit,
            )
            summaries.append(summary)
        aggregate = runner.aggregate_user_summaries(
            data_dir=args.data_dir,
            out_root=stride_root,
            user_ids=user_ids,
            summaries=summaries,
        )
        aggregate["observation_density"] = {
            "response_stride_steps": stride,
            "response_update_interval_minutes": interval_minutes(stride),
            "evaluation_grid": "five_second_full_day",
            "carry_forward": "target_response_weights",
        }
        summary_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
        runner.write_confusion_matrix_csv(
            stride_root / "pooled_confusion_matrix.csv",
            np.asarray(aggregate["pooled_confusion_matrix"], dtype=np.int64),
        )
        runner.write_aggregate_markdown_report(stride_root / "aggregate_report.md", aggregate)
        aggregates_by_stride[stride] = aggregate

    method_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    if args.full_baseline_root and (args.full_baseline_root / "all_users_summary.json").exists():
        full_aggregate = load_aggregate(args.full_baseline_root)
        method_rows.append(
            method_summary_row(
                label=density_label(1),
                stride_steps=1,
                aggregate=full_aggregate,
                result_root=args.full_baseline_root,
            )
        )
        class_rows.extend(
            class_metric_rows(label=density_label(1), stride_steps=1, aggregate=full_aggregate)
        )

    for stride in strides:
        aggregate = aggregates_by_stride[stride]
        method_rows.append(
            method_summary_row(
                label=density_label(stride),
                stride_steps=stride,
                aggregate=aggregate,
                result_root=result_roots_by_stride[stride],
            )
        )
        class_rows.extend(class_metric_rows(label=density_label(stride), stride_steps=stride, aggregate=aggregate))

    write_csv(args.out_root / "target_response_density_method_summary.csv", method_rows)
    write_csv(args.out_root / "table2_target_response_density.csv", class_rows)
    write_latex_rows(args.out_root / "table2_target_response_density_latex_rows.tex", class_rows)
    write_summary_markdown(args.out_root / "summary.md", method_rows)
    (args.out_root / "run_config.json").write_text(
        json.dumps(
            {
                "code_root": str(args.code_root),
                "data_dir": str(args.data_dir),
                "out_root": str(args.out_root),
                "full_baseline_root": str(args.full_baseline_root),
                "user_ids": user_ids,
                "response_strides": strides,
                "storage_dtype": args.storage_dtype,
                "device": args.device,
                "torch_dtype": args.torch_dtype,
                "target_day_batch_size": args.target_day_batch_size,
                "history_chunk_size": args.history_chunk_size,
                "reference_sampling": {
                    "mode": reference_sampling_config.mode,
                    "sample_size": reference_sampling_config.sample_size,
                    "num_trials": reference_sampling_config.num_trials,
                    "seed": reference_sampling_config.seed,
                    "history_window": reference_sampling_config.history_window,
                },
                "params_hat": {
                    "rho": args.params_hat_rho,
                    "lambda_": args.params_hat_lambda,
                    "alpha_mix": list(params_hat_alpha_mix),
                },
                "time_step_limit": args.time_step_limit,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"method_summary": method_rows}, indent=2), flush=True)

if __name__ == "__main__":
    main()
