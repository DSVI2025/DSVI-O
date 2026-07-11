from __future__ import annotations

import csv
import importlib.util
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
RUNNER_PATH = ROOT / "run_old_matlab_equivalent.py"
RUNNER_MODULE_NAME = "_transfer_runner_old_matlab_equivalent"

DEFAULT_BASELINE_ROOT = REPO_ROOT / "results" / "s6_e2_target_full_recomputation"
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "mat" / "source_target_shared_emr"
DEFAULT_CACHE_ROOT = (
    REPO_ROOT / "results" / "s6_e5_similarity_weighted_topk" / "source_response_cache"
)
DEFAULT_TRANSFER_ROOT = REPO_ROOT / "results" / "s6_e5_similarity_weighted_topk"

DEFAULT_PARAMS_HAT_RHO = 0.7
DEFAULT_PARAMS_HAT_LAMBDA = 0.305
DEFAULT_PARAMS_HAT_ALPHA_MIX = (0.75, 0.58, 0.16)
DEFAULT_HISTORY_WINDOW = 12
DEFAULT_REFERENCE_SAMPLE_SIZE = 50
DEFAULT_REFERENCE_NUM_TRIALS = 20
DEFAULT_REFERENCE_SEED = 20260423
DEFAULT_TARGET_DAY_BATCH_SIZE = 10
DEFAULT_HISTORY_CHUNK_SIZE = 1024
DEFAULT_STORAGE_DTYPE = "float32"
DEFAULT_TORCH_DTYPE = "float64"
DEFAULT_DEVICE = "cuda"
DEFAULT_PROGRESS_INTERVAL = 300
DEFAULT_USER_IDS = tuple(range(1, 11))

@dataclass(frozen=True)
class TransferDefaults:
    baseline_root: Path = DEFAULT_BASELINE_ROOT
    data_dir: Path = DEFAULT_DATA_DIR
    cache_root: Path = DEFAULT_CACHE_ROOT
    transfer_root: Path = DEFAULT_TRANSFER_ROOT
    params_hat_rho: float = DEFAULT_PARAMS_HAT_RHO
    params_hat_lambda: float = DEFAULT_PARAMS_HAT_LAMBDA
    params_hat_alpha_mix: tuple[float, float, float] = DEFAULT_PARAMS_HAT_ALPHA_MIX
    history_window: int = DEFAULT_HISTORY_WINDOW
    reference_sample_size: int = DEFAULT_REFERENCE_SAMPLE_SIZE
    reference_num_trials: int = DEFAULT_REFERENCE_NUM_TRIALS
    reference_seed: int = DEFAULT_REFERENCE_SEED
    target_day_batch_size: int = DEFAULT_TARGET_DAY_BATCH_SIZE
    history_chunk_size: int = DEFAULT_HISTORY_CHUNK_SIZE
    storage_dtype: str = DEFAULT_STORAGE_DTYPE
    torch_dtype: str = DEFAULT_TORCH_DTYPE
    device: str = DEFAULT_DEVICE
    progress_interval: int = DEFAULT_PROGRESS_INTERVAL

DEFAULTS = TransferDefaults()

def load_runner_module():
    if RUNNER_MODULE_NAME in sys.modules:
        return sys.modules[RUNNER_MODULE_NAME]

    spec = importlib.util.spec_from_file_location(RUNNER_MODULE_NAME, RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load runner module from {RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[RUNNER_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module

def parse_int_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]

def parse_k_values(raw: str) -> list[int]:
    values = sorted({int(item.strip()) for item in raw.split(",") if item.strip()})
    if not values:
        raise ValueError("At least one K value is required.")
    return values

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        keys: set[str] = set()
        for row in rows:
            keys.update(row.keys())
        fieldnames = sorted(keys)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

def serialize_float_list(values: list[float], precision: int = 10) -> str:
    return ";".join(f"{float(value):.{precision}g}" for value in values)

def parse_storage_dtype(runner, name: str):
    return runner.resolve_storage_dtype(name)

def parse_torch_dtype(runner, name: str):
    return runner.resolve_torch_dtype(name)

def resolve_device(runner, name: str):
    return runner.resolve_torch_device(name)

def make_reference_sampling_config(runner):
    return runner.ReferenceSamplingConfig(
        mode="sampled",
        sample_size=DEFAULTS.reference_sample_size,
        num_trials=DEFAULTS.reference_num_trials,
        seed=DEFAULTS.reference_seed,
        history_window=DEFAULTS.history_window,
    )

def make_params_true(runner):
    return runner.ParamsTrue(
        alpha_ou=float(np.exp(-4.0)),
        A=(0.15, 0.17),
        f=(0.025, 0.021),
        sigma=0.5,
        theta=(0.5, 1.5),
    )

def make_params_hat(runner):
    return runner.ParamsHat(
        alpha_ou=float(np.exp(-4.0)),
        A=(0.11, 0.13),
        f=(0.035, 0.038),
        sigma=0.4,
        theta=(0.5, 1.5),
        rho=DEFAULTS.params_hat_rho,
        lambda_=DEFAULTS.params_hat_lambda,
        alpha_mix=DEFAULTS.params_hat_alpha_mix,
    )

def baseline_reference_sampling_payload() -> dict[str, Any]:
    return {
        "mode": "sampled",
        "sample_size": DEFAULTS.reference_sample_size,
        "num_trials": DEFAULTS.reference_num_trials,
        "seed": DEFAULTS.reference_seed,
        "history_window": DEFAULTS.history_window,
    }

def transfer_noise_payload() -> dict[str, Any]:
    return {
        "preset": "none",
        "seed": DEFAULTS.reference_seed,
        "component_mode": "none",
        "strength_multiplier": 0.0,
        "apply_scope": "all",
    }

def cache_user_dir(cache_root: Path, user_id: int) -> Path:
    return cache_root / f"user{user_id:02d}"

def load_response_cache(cache_root: Path, user_id: int) -> dict[str, np.ndarray]:
    cache_path = cache_user_dir(cache_root, user_id) / "response_cache.npz"
    if not cache_path.exists():
        raise FileNotFoundError(f"Missing response cache: {cache_path}")
    data = np.load(cache_path, allow_pickle=False)
    return {key: data[key] for key in data.files}

def flatten_metric_record(summary: dict[str, Any]) -> dict[str, Any]:
    transfer = dict(summary.get("transfer", {}))
    source_user_ids = [int(value) for value in transfer.get("source_user_ids", [])]
    source_distances = [float(value) for value in transfer.get("source_distances", [])]
    weights = [float(value) for value in transfer.get("weights", [])]
    return {
        "user_id": int(summary["user_id"]),
        "method": transfer.get("method", ""),
        "k": int(transfer.get("k", 0)),
        "tau": float(transfer.get("tau", 0.0)) if transfer.get("tau") is not None else "",
        "source_user_ids": ",".join(f"{value:02d}" for value in source_user_ids),
        "source_distances": serialize_float_list(source_distances),
        "weights": serialize_float_list(weights),
        "accuracy_percent": float(summary["accuracy_percent"]),
        "macro_f1": float(summary["macro_f1"]),
        "balanced_accuracy": float(summary["balanced_accuracy"]),
        "weighted_f1": float(summary["weighted_f1"]),
        "worst_class_recall": float(summary["worst_class_recall"]),
        "worst_class_f1": float(summary["worst_class_f1"]),
        "target_days_runtime_seconds": float(summary["target_days_runtime_seconds"]),
    }

def aggregate_transfer_summaries(
    runner,
    *,
    data_dir: Path,
    out_root: Path,
    user_ids: list[int],
    summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    aggregate = runner.aggregate_user_summaries(
        data_dir=data_dir,
        out_root=out_root,
        user_ids=user_ids,
        summaries=summaries,
    )
    aggregate["noise"] = transfer_noise_payload()
    aggregate["reference_sampling"] = baseline_reference_sampling_payload()
    return aggregate

def build_similarity_feature_rows(
    runner,
    *,
    data_dir: Path,
    user_ids: list[int],
    storage_dtype_name: str = DEFAULT_STORAGE_DTYPE,
) -> tuple[list[dict[str, Any]], dict[int, dict[str, np.ndarray]]]:
    storage_dtype = parse_storage_dtype(runner, storage_dtype_name)
    reference_idx = runner.OTHERS.astype(np.int64) - 1
    rows: list[dict[str, Any]] = []
    features_by_user: dict[int, dict[str, np.ndarray]] = {}

    for user_id in user_ids:
        h_all, _labels_all = runner.load_user_tensors(data_dir, user_id, storage_dtype)
        h1 = np.asarray(h_all[0][reference_idx], dtype=np.float64)
        h2 = np.asarray(h_all[1][reference_idx], dtype=np.float64)
        h3 = np.asarray(h_all[2], dtype=np.float64)

        emr = h3.mean(axis=0)
        watch_mean = h1.mean(axis=(0, 2))
        watch_std = h1.std(axis=(0, 2))
        insole_mean = h2.mean(axis=(0, 2))
        insole_std = h2.std(axis=(0, 2))

        watch = np.concatenate([watch_mean, watch_std], axis=0)
        insole = np.concatenate([insole_mean, insole_std], axis=0)
        features_by_user[user_id] = {
            "emr": emr,
            "watch": watch,
            "insole": insole,
        }
        rows.append(
            {
                "user_id": user_id,
                "emr": emr,
                "watch": watch,
                "insole": insole,
            }
        )
    return rows, features_by_user

def compute_pairwise_similarity(
    runner,
    *,
    data_dir: Path,
    user_ids: list[int],
    storage_dtype_name: str = DEFAULT_STORAGE_DTYPE,
) -> tuple[list[dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    _rows, features_by_user = build_similarity_feature_rows(
        runner,
        data_dir=data_dir,
        user_ids=user_ids,
        storage_dtype_name=storage_dtype_name,
    )
    emr_stack = np.stack([features_by_user[user_id]["emr"] for user_id in user_ids], axis=0)
    watch_stack = np.stack([features_by_user[user_id]["watch"] for user_id in user_ids], axis=0)
    insole_stack = np.stack([features_by_user[user_id]["insole"] for user_id in user_ids], axis=0)

    emr_mu = emr_stack.mean(axis=0)
    emr_sigma = np.maximum(emr_stack.std(axis=0), 1e-12)
    watch_mu = watch_stack.mean(axis=0)
    watch_sigma = np.maximum(watch_stack.std(axis=0), 1e-12)
    insole_mu = insole_stack.mean(axis=0)
    insole_sigma = np.maximum(insole_stack.std(axis=0), 1e-12)

    standardized: dict[int, dict[str, np.ndarray]] = {}
    for user_id in user_ids:
        standardized[user_id] = {
            "emr": (features_by_user[user_id]["emr"] - emr_mu) / emr_sigma,
            "watch": (features_by_user[user_id]["watch"] - watch_mu) / watch_sigma,
            "insole": (features_by_user[user_id]["insole"] - insole_mu) / insole_sigma,
        }

    pair_rows: list[dict[str, Any]] = []
    ranking_by_target: dict[int, list[dict[str, Any]]] = {}
    for target_user_id in user_ids:
        target_rows: list[dict[str, Any]] = []
        target_features = standardized[target_user_id]
        for source_user_id in user_ids:
            if source_user_id == target_user_id:
                continue
            source_features = standardized[source_user_id]
            d_emr = float(np.sum(np.square(target_features["emr"] - source_features["emr"])))
            d_watch = float(np.sum(np.square(target_features["watch"] - source_features["watch"])))
            d_insole = float(
                np.sum(np.square(target_features["insole"] - source_features["insole"]))
            )
            distance = float(d_emr + 2.0 * d_watch + 2.0 * d_insole)
            row = {
                "target_user_id": target_user_id,
                "source_user_id": source_user_id,
                "distance": distance,
                "d_emr": d_emr,
                "d_watch": d_watch,
                "d_insole": d_insole,
            }
            pair_rows.append(row)
            target_rows.append(row)
        target_rows.sort(key=lambda item: (float(item["distance"]), int(item["source_user_id"])))
        for rank, row in enumerate(target_rows, start=1):
            row["rank"] = rank
        ranking_by_target[target_user_id] = target_rows
    pair_rows.sort(
        key=lambda item: (
            int(item["target_user_id"]),
            float(item["distance"]),
            int(item["source_user_id"]),
        )
    )
    return pair_rows, ranking_by_target

def compute_source_target_similarity(
    runner,
    *,
    data_dir: Path,
    target_user_ids: list[int],
    source_user_ids: list[int],
    storage_dtype_name: str = DEFAULT_STORAGE_DTYPE,
) -> tuple[list[dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    all_user_ids = sorted(set(target_user_ids) | set(source_user_ids))
    _rows, features_by_user = build_similarity_feature_rows(
        runner,
        data_dir=data_dir,
        user_ids=all_user_ids,
        storage_dtype_name=storage_dtype_name,
    )
    emr_stack = np.stack([features_by_user[user_id]["emr"] for user_id in all_user_ids], axis=0)
    watch_stack = np.stack([features_by_user[user_id]["watch"] for user_id in all_user_ids], axis=0)
    insole_stack = np.stack([features_by_user[user_id]["insole"] for user_id in all_user_ids], axis=0)

    emr_mu = emr_stack.mean(axis=0)
    emr_sigma = np.maximum(emr_stack.std(axis=0), 1e-12)
    watch_mu = watch_stack.mean(axis=0)
    watch_sigma = np.maximum(watch_stack.std(axis=0), 1e-12)
    insole_mu = insole_stack.mean(axis=0)
    insole_sigma = np.maximum(insole_stack.std(axis=0), 1e-12)

    standardized: dict[int, dict[str, np.ndarray]] = {}
    for user_id in all_user_ids:
        standardized[user_id] = {
            "emr": (features_by_user[user_id]["emr"] - emr_mu) / emr_sigma,
            "watch": (features_by_user[user_id]["watch"] - watch_mu) / watch_sigma,
            "insole": (features_by_user[user_id]["insole"] - insole_mu) / insole_sigma,
        }

    pair_rows: list[dict[str, Any]] = []
    ranking_by_target: dict[int, list[dict[str, Any]]] = {}
    for target_user_id in target_user_ids:
        target_rows: list[dict[str, Any]] = []
        target_features = standardized[target_user_id]
        for source_user_id in source_user_ids:
            if source_user_id == target_user_id:
                continue
            source_features = standardized[source_user_id]
            d_emr = float(np.sum(np.square(target_features["emr"] - source_features["emr"])))
            d_watch = float(np.sum(np.square(target_features["watch"] - source_features["watch"])))
            d_insole = float(
                np.sum(np.square(target_features["insole"] - source_features["insole"]))
            )
            distance = float(d_emr + 2.0 * d_watch + 2.0 * d_insole)
            row = {
                "target_user_id": target_user_id,
                "source_user_id": source_user_id,
                "distance": distance,
                "d_emr": d_emr,
                "d_watch": d_watch,
                "d_insole": d_insole,
            }
            pair_rows.append(row)
            target_rows.append(row)
        target_rows.sort(key=lambda item: (float(item["distance"]), int(item["source_user_id"])))
        for rank, row in enumerate(target_rows, start=1):
            row["rank"] = rank
        ranking_by_target[target_user_id] = target_rows
    pair_rows.sort(
        key=lambda item: (
            int(item["target_user_id"]),
            float(item["distance"]),
            int(item["source_user_id"]),
        )
    )
    return pair_rows, ranking_by_target

def build_similarity_csv_rows(pair_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "target_user_id": int(row["target_user_id"]),
            "source_user_id": int(row["source_user_id"]),
            "rank": int(row["rank"]),
            "distance": float(row["distance"]),
            "d_emr": float(row["d_emr"]),
            "d_watch": float(row["d_watch"]),
            "d_insole": float(row["d_insole"]),
        }
        for row in pair_rows
    ]

def run_transfer_inference_for_user(
    runner,
    *,
    data_dir: Path,
    user_id: int,
    transferred_y1: np.ndarray,
    transferred_y2: np.ndarray,
    transferred_y3: np.ndarray,
    method: str,
    source_user_ids: list[int],
    source_distances: list[float],
    weights: list[float],
    tau: float | None,
    k: int,
    storage_dtype_name: str = DEFAULT_STORAGE_DTYPE,
    time_step_limit: int | None = None,
) -> dict[str, Any]:
    storage_dtype = parse_storage_dtype(runner, storage_dtype_name)
    h_all, labels_all = runner.load_user_tensors(data_dir, user_id, storage_dtype)
    params_true = make_params_true(runner)
    params_hat = make_params_hat(runner)
    day_idx = runner.DAYS.astype(np.int64) - 1

    h1_days = np.asarray(h_all[0][day_idx], dtype=np.float64)
    h2_days = np.asarray(h_all[1][day_idx], dtype=np.float64)
    h3_days = np.asarray(h_all[2][day_idx], dtype=np.float64)
    b_days = np.asarray(labels_all[day_idx], dtype=np.float64)

    num_days = int(len(runner.DAYS))
    t_day_full = int(labels_all.shape[1])
    transferred_t_day = int(transferred_y1.shape[1])
    t_day = t_day_full
    if time_step_limit is not None:
        t_day = min(t_day, int(time_step_limit))
    t_day = min(t_day, transferred_t_day)
    if t_day <= 0:
        raise ValueError("time_step_limit must be positive when provided.")
    x = np.zeros((num_days, t_day + 1), dtype=np.float64)
    xhat = np.zeros((num_days, t_day + 1), dtype=np.float64)
    xhat[:, 0] = 0.5
    theta_true = np.asarray(params_true.theta, dtype=np.float64)
    started = time.time()

    for t in range(t_day):
        m_t = b_days[:, t]
        s_true = 0.5 * (1.0 + np.tanh((m_t[:, None] - theta_true[None, :]) / params_true.sigma))
        ell_true = np.sum(s_true, axis=1)
        q_true = (
            params_true.A[0] * math.sin(2.0 * math.pi * params_true.f[0] * t)
            + params_true.A[1] * math.sin(2.0 * math.pi * params_true.f[1] * t)
        )
        x[:, t + 1] = np.clip(
            params_true.alpha_ou * x[:, t] + (1.0 - params_true.alpha_ou) * (ell_true + q_true),
            0.0,
            2.0,
        )

        mhat = (
            params_hat.alpha_mix[0] * np.sum(transferred_y1[:, t, :] * h1_days[:, :, t], axis=1)
            + params_hat.alpha_mix[1] * np.sum(transferred_y2[:, t, :] * h2_days[:, :, t], axis=1)
            + params_hat.alpha_mix[2] * np.sum(transferred_y3[:, t, :] * h3_days, axis=1)
        )
        q_hat = (
            params_hat.A[0] * math.sin(2.0 * math.pi * params_hat.f[0] * t)
            + params_hat.A[1] * math.sin(2.0 * math.pi * params_hat.f[1] * t)
        )
        xhat[:, t + 1] = np.clip(
            params_hat.alpha_ou * xhat[:, t] + (1.0 - params_hat.alpha_ou) * (mhat + q_hat),
            0.0,
            2.0,
        )

    target_days_runtime_seconds = time.time() - started
    x_sm = np.stack([runner.smooth_gaussian_same(row[1:], window=21) for row in x], axis=0)
    xhat_sm = np.stack([runner.smooth_gaussian_same(row[1:], window=21) for row in xhat], axis=0)
    true_disc = runner.discretize_three_level(x_sm)
    pred_disc = runner.discretize_three_level(xhat_sm)
    confusion = runner.confusion_matrix_3(true_disc, pred_disc)
    metrics = runner.classification_metrics_from_confusion_matrix(confusion)

    day_summaries: list[dict[str, Any]] = []
    for local_idx, day_value in enumerate(runner.DAYS.tolist()):
        day_confusion = runner.confusion_matrix_3(true_disc[local_idx], pred_disc[local_idx])
        day_metrics = runner.classification_metrics_from_confusion_matrix(day_confusion)
        day_summaries.append(
            {
                "user_id": user_id,
                "day_order": local_idx + 1,
                "num_days": num_days,
                "day_1based": int(day_value),
                "t_day": t_day,
                "accuracy_percent": day_metrics["accuracy"] * 100.0,
                "confusion_matrix": day_confusion.tolist(),
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
                "runtime_seconds": target_days_runtime_seconds / float(num_days),
                "runtime_minutes": target_days_runtime_seconds / float(num_days) / 60.0,
                "runtime_note": "transfer replay mean per day",
            }
        )

    return {
        "user_id": user_id,
        "days_1based": runner.DAYS.tolist(),
        "sample_days_1based": runner.OTHERS.tolist(),
        "reference_sampling": baseline_reference_sampling_payload(),
        "t_day": t_day,
        "t_day_full": t_day_full,
        "runtime_seconds": target_days_runtime_seconds,
        "runtime_minutes": target_days_runtime_seconds / 60.0,
        "target_days_runtime_seconds": target_days_runtime_seconds,
        "target_days_runtime_minutes": target_days_runtime_seconds / 60.0,
        "accuracy_percent": metrics["accuracy"] * 100.0,
        "confusion_matrix": confusion.tolist(),
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
        "noise": transfer_noise_payload(),
        "compute": {
            "backend": "transfer_cached_response",
            "device": "cpu",
            "torch_dtype": DEFAULTS.torch_dtype,
            "target_day_batch_size": DEFAULTS.target_day_batch_size,
            "history_chunk_size": DEFAULTS.history_chunk_size,
            "verify_against_cpu": False,
            "time_step_limit": time_step_limit,
        },
        "model": {
            "params_hat": {
                "rho": params_hat.rho,
                "lambda_": params_hat.lambda_,
                "alpha_mix": list(params_hat.alpha_mix),
            }
        },
        "transfer": {
            "method": method,
            "k": int(k),
            "tau": None if tau is None else float(tau),
            "source_user_ids": [int(value) for value in source_user_ids],
            "source_distances": [float(value) for value in source_distances],
            "weights": [float(value) for value in weights],
        },
    }

def average_confusions(confusions: list[np.ndarray]) -> np.ndarray:
    if not confusions:
        raise ValueError("No confusion matrices provided.")
    return np.mean(np.stack(confusions, axis=0), axis=0)

def build_random_source_summary(method1_pair_records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for record in method1_pair_records:
        grouped.setdefault(int(record["user_id"]), []).append(record)
    user_summaries: list[dict[str, Any]] = []
    for user_id, records in sorted(grouped.items()):
        confusion_avg = average_confusions(
            [np.asarray(record["confusion_matrix"], dtype=np.float64) for record in records]
        )
        confusion_int = np.rint(confusion_avg).astype(np.int64)
        runtime_seconds = float(
            np.mean([float(record["target_days_runtime_seconds"]) for record in records], dtype=np.float64)
        )
        metrics = load_runner_module().classification_metrics_from_confusion_matrix(confusion_int)
        user_summaries.append(
            {
                "user_id": user_id,
                "accuracy_percent": metrics["accuracy"] * 100.0,
                "macro_f1": metrics["macro_f1"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "weighted_f1": metrics["weighted_f1"],
                "worst_class_recall": metrics["worst_class_recall"],
                "worst_class_f1": metrics["worst_class_f1"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "specificity": metrics["specificity"],
                "f1": metrics["f1"],
                "support": metrics["support"],
                "confusion_matrix": confusion_int.tolist(),
                "runtime_seconds": runtime_seconds,
                "runtime_minutes": runtime_seconds / 60.0,
                "target_days_runtime_seconds": runtime_seconds,
                "target_days_runtime_minutes": runtime_seconds / 60.0,
                "reference_sampling": baseline_reference_sampling_payload(),
                "noise": transfer_noise_payload(),
                "transfer": {
                    "method": "random_source_expected",
                    "k": 1,
                    "tau": None,
                    "source_user_ids": sorted(
                        int(record["transfer"]["source_user_ids"][0]) for record in records
                    ),
                    "source_distances": sorted(
                        float(record["transfer"]["source_distances"][0]) for record in records
                    ),
                    "weights": [1.0 / float(len(records))] * len(records),
                },
            }
        )
    pooled_confusion = np.sum(
        np.stack([np.asarray(summary["confusion_matrix"], dtype=np.int64) for summary in user_summaries], axis=0),
        axis=0,
    )
    runner = load_runner_module()
    pooled_metrics = runner.classification_metrics_from_confusion_matrix(pooled_confusion)
    return {
        "method": "random_source_expected",
        "num_users": len(user_summaries),
        "summaries": user_summaries,
        "pooled_confusion_matrix": pooled_confusion.tolist(),
        "pooled_metrics": {
            "accuracy_percent": pooled_metrics["accuracy"] * 100.0,
            "macro_precision": pooled_metrics["macro_precision"],
            "macro_recall": pooled_metrics["macro_recall"],
            "macro_f1": pooled_metrics["macro_f1"],
            "balanced_accuracy": pooled_metrics["balanced_accuracy"],
            "weighted_f1": pooled_metrics["weighted_f1"],
            "worst_class_recall": pooled_metrics["worst_class_recall"],
            "worst_class_f1": pooled_metrics["worst_class_f1"],
            "precision": pooled_metrics["precision"],
            "recall": pooled_metrics["recall"],
            "specificity": pooled_metrics["specificity"],
            "f1": pooled_metrics["f1"],
            "support": pooled_metrics["support"],
        },
        "target_days_runtime_seconds_mean": float(
            np.mean([float(summary["target_days_runtime_seconds"]) for summary in user_summaries], dtype=np.float64)
        ),
    }
