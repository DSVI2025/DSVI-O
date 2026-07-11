from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from _transfer_learning_common import (
    DEFAULTS,
    cache_user_dir,
    ensure_dir,
    load_runner_module,
    make_params_hat,
    make_reference_sampling_config,
    parse_int_list,
    parse_storage_dtype,
    parse_torch_dtype,
    resolve_device,
    write_json,
)

def export_user_cache(
    runner,
    *,
    data_dir: Path,
    output_root: Path,
    user_id: int,
    storage_dtype_name: str,
    torch_dtype_name: str,
    device_name: str,
    history_chunk_size: int,
    progress_interval: int,
    baseline_root: Path,
    time_step_limit: int | None,
) -> dict[str, object]:
    storage_dtype = parse_storage_dtype(runner, storage_dtype_name)
    torch_dtype = parse_torch_dtype(runner, torch_dtype_name)
    device = resolve_device(runner, device_name)
    params_hat = make_params_hat(runner)
    reference_sampling_config = make_reference_sampling_config(runner)
    reference_sampling_config = runner.effective_reference_sampling_config(
        reference_sampling_config,
        reference_pool_size=len(runner.OTHERS),
    )

    h_all, labels_all = runner.load_user_tensors(data_dir, user_id, storage_dtype)
    t_day_full = int(labels_all.shape[1])
    t_day = t_day_full if time_step_limit is None else min(int(time_step_limit), t_day_full)
    if t_day <= 0:
        raise ValueError("time_step_limit must be positive when provided.")
    day_idx = torch.as_tensor(runner.DAYS.astype(np.int64) - 1, device=device, dtype=torch.long)
    h_all_torch = [torch.as_tensor(arr, device=device, dtype=torch_dtype) for arr in h_all]
    labels_all_torch = torch.as_tensor(labels_all, device=device, dtype=torch_dtype)

    sample_idx = runner.OTHERS.astype(np.int64) - 1
    reference_trial_indices = runner.build_reference_trial_indices(
        sample_idx=sample_idx,
        config=reference_sampling_config,
        user_id=user_id,
    )
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

    b3 = h_all_torch[2][day_idx, :]
    hsamp3 = h_all_torch[2][reference_trial_idx_torch]
    csamp3 = torch.mean(labels_all_torch[reference_trial_idx_torch], dim=2)
    h3_th, h3_tc, h3_tn = runner.precompute_static_modal_terms_days_torch(
        b=b3,
        hsamp=hsamp3,
        csamp=csamp3,
        params_hat=params_hat,
    )

    num_days = len(runner.DAYS)
    xhat = torch.zeros((num_days, t_day + 1), device=device, dtype=torch_dtype)
    xhat[:, 0] = 0.5
    y1_mean = np.empty((num_days, t_day, h_all[0].shape[1]), dtype=np.float64)
    y2_mean = np.empty((num_days, t_day, h_all[1].shape[1]), dtype=np.float64)
    y3_mean = np.empty((num_days, t_day, h_all[2].shape[1]), dtype=np.float64)

    started = time.time()
    for t in range(t_day):
        b1 = h_all_torch[0][day_idx, :, t]
        b2 = h_all_torch[1][day_idx, :, t]
        y1_trials = runner.solve_history_modal_y_days_torch(
            b=b1,
            xhat_t=xhat[:, t],
            cache=history_precomputes[0],
            current_t=t,
            params_hat=params_hat,
            history_chunk_size=history_chunk_size,
        )
        y2_trials = runner.solve_history_modal_y_days_torch(
            b=b2,
            xhat_t=xhat[:, t],
            cache=history_precomputes[1],
            current_t=t,
            params_hat=params_hat,
            history_chunk_size=history_chunk_size,
        )
        y3_trials = runner.solve_static_modal_y_days_torch(
            b=b3,
            h_th=h3_th,
            h_tc=h3_tc,
            h_tn=h3_tn,
            xhat_t=xhat[:, t],
            params_hat=params_hat,
        )
        y1_mean_t = torch.mean(y1_trials, dim=1)
        y2_mean_t = torch.mean(y2_trials, dim=1)
        y3_mean_t = torch.mean(y3_trials, dim=1)

        y1_mean[:, t, :] = y1_mean_t.detach().cpu().numpy()
        y2_mean[:, t, :] = y2_mean_t.detach().cpu().numpy()
        y3_mean[:, t, :] = y3_mean_t.detach().cpu().numpy()

        mhat_trials = (
            params_hat.alpha_mix[0] * torch.sum(y1_trials * b1[:, None, :], dim=2)
            + params_hat.alpha_mix[1] * torch.sum(y2_trials * b2[:, None, :], dim=2)
            + params_hat.alpha_mix[2] * torch.sum(y3_trials * b3[:, None, :], dim=2)
        )
        ell_hat = torch.mean(mhat_trials, dim=1)
        q_hat = (
            params_hat.A[0] * np.sin(2.0 * np.pi * params_hat.f[0] * t)
            + params_hat.A[1] * np.sin(2.0 * np.pi * params_hat.f[1] * t)
        )
        xhat[:, t + 1] = torch.clamp(
            params_hat.alpha_ou * xhat[:, t] + (1.0 - params_hat.alpha_ou) * (ell_hat + q_hat),
            min=0.0,
            max=2.0,
        )
        if progress_interval > 0 and (
            t == 0 or (t + 1) % progress_interval == 0 or (t + 1) == t_day
        ):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.time() - started
            print(
                f"[cache user {user_id:02d}] t={t + 1}/{t_day} "
                f"xhat_mean={float(torch.mean(xhat[:, t + 1]).detach().cpu()):.6f} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.time() - started

    user_out_dir = cache_user_dir(output_root, user_id)
    ensure_dir(user_out_dir)
    np.savez_compressed(
        user_out_dir / "response_cache.npz",
        days=runner.DAYS.astype(np.int64),
        y1_mean=y1_mean,
        y2_mean=y2_mean,
        y3_mean=y3_mean,
    )
    manifest = {
        "user_id": user_id,
        "baseline_root": str(baseline_root),
        "data_dir": str(data_dir),
        "days_1based": runner.DAYS.tolist(),
        "reference_sampling": {
            "mode": reference_sampling_config.mode,
            "sample_size": reference_sampling_config.sample_size,
            "num_trials": reference_sampling_config.num_trials,
            "seed": reference_sampling_config.seed,
            "history_window": reference_sampling_config.history_window,
        },
        "params_hat": {
            "rho": params_hat.rho,
            "lambda_": params_hat.lambda_,
            "alpha_mix": list(params_hat.alpha_mix),
        },
        "compute": {
            "device": str(device),
            "storage_dtype": storage_dtype_name,
            "torch_dtype": torch_dtype_name,
            "history_chunk_size": history_chunk_size,
            "time_step_limit": time_step_limit,
        },
        "shapes": {
            "y1_mean": list(y1_mean.shape),
            "y2_mean": list(y2_mean.shape),
            "y3_mean": list(y3_mean.shape),
        },
        "t_day_full": t_day_full,
        "t_day_exported": t_day,
        "runtime_seconds": elapsed,
        "generated_at_unix": time.time(),
    }
    write_json(user_out_dir / "cache_manifest.json", manifest)
    return manifest

def main() -> None:
    runner = load_runner_module()
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULTS.data_dir)
    parser.add_argument("--output-root", type=Path, default=DEFAULTS.cache_root)
    parser.add_argument("--baseline-root", type=Path, default=DEFAULTS.baseline_root)
    parser.add_argument(
        "--user-ids",
        default="1,2,3,4,5,6,7,8,9,10",
        help="Comma-separated user ids to export.",
    )
    parser.add_argument("--storage-dtype", default=DEFAULTS.storage_dtype, choices=["float32", "float64"])
    parser.add_argument("--torch-dtype", default=DEFAULTS.torch_dtype, choices=["float32", "float64"])
    parser.add_argument("--device", default=DEFAULTS.device)
    parser.add_argument("--history-chunk-size", type=int, default=DEFAULTS.history_chunk_size)
    parser.add_argument("--progress-interval", type=int, default=DEFAULTS.progress_interval)
    parser.add_argument(
        "--time-step-limit",
        type=int,
        default=None,
        help="Optional smoke-test limit on the number of time steps exported per day.",
    )
    args = parser.parse_args()

    if args.history_chunk_size <= 0:
        raise ValueError("--history-chunk-size must be positive.")
    if args.time_step_limit is not None and args.time_step_limit <= 0:
        raise ValueError("--time-step-limit must be positive when provided.")

    user_ids = parse_int_list(args.user_ids)
    ensure_dir(args.output_root)
    runner.configure_torch_runtime()

    manifests = []
    for user_id in user_ids:
        print(f"=== Exporting response cache for user {user_id:02d} ===", flush=True)
        manifest = export_user_cache(
            runner,
            data_dir=args.data_dir,
            output_root=args.output_root,
            user_id=user_id,
            storage_dtype_name=args.storage_dtype,
            torch_dtype_name=args.torch_dtype,
            device_name=args.device,
            history_chunk_size=args.history_chunk_size,
            progress_interval=args.progress_interval,
            baseline_root=args.baseline_root,
            time_step_limit=args.time_step_limit,
        )
        manifests.append(manifest)

    write_json(
        args.output_root / "export_manifest.json",
        {
            "baseline_root": str(args.baseline_root),
            "data_dir": str(args.data_dir),
            "user_ids": user_ids,
            "num_users": len(user_ids),
            "items": manifests,
        },
    )

if __name__ == "__main__":
    main()
