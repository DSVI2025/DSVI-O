from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import windows
import torch

DAYS = np.arange(91, 101, dtype=np.int64)
OTHERS = np.arange(1, 91, dtype=np.int64)
H1_FEATURES = [
    "heart_rate",
    "spo2",
    "sleep_quality",
    "steps",
    "calories",
    "skin_temperature",
    "hrv",
    "stress_index",
    "sbp",
    "dbp",
    "activity_intensity",
    "active_minutes",
    "floors_climbed",
    "activity",
]
H2_FEATURES = [
    "LF",
    "LH",
    "RF",
    "RH",
    "step_freq",
    "stride_length",
    "contact_time",
    "symmetry",
    "gait_type",
    "motion_type",
    "force_estimate",
    "balance_score",
    "lr_ratio",
    "center_x",
    "center_y",
    "exercise_intensity",
    "plantar_fatigue",
]
H1_DISCRETE_FEATURES = {"sleep_quality", "activity"}
H2_DISCRETE_FEATURES = {"gait_type", "motion_type"}
H1_CUMULATIVE_FEATURES = {"steps", "active_minutes", "floors_climbed"}
H1_ADDITIVE_DRIFT_FEATURES = {
    "heart_rate",
    "spo2",
    "calories",
    "skin_temperature",
    "hrv",
    "stress_index",
    "sbp",
    "dbp",
    "activity_intensity",
}
H2_MULTIPLICATIVE_FEATURES = {
    "LF",
    "LH",
    "RF",
    "RH",
    "step_freq",
    "stride_length",
    "contact_time",
    "force_estimate",
    "exercise_intensity",
    "plantar_fatigue",
}
H2_RATIO_FEATURES = {"symmetry", "balance_score", "lr_ratio"}
H2_CENTER_FEATURES = {"center_x", "center_y"}
H1_LAPLACE_FEATURES = {"spo2"}
H1_IMPULSE_FEATURES: set[str] = set()
H2_LAPLACE_FEATURES = {"force_estimate", "center_x", "center_y"}
H2_IMPULSE_FEATURES = {"LF", "LH", "RF", "RH", "force_estimate", "center_x", "center_y"}

@dataclass(frozen=True)
class ParamsTrue:
    alpha_ou: float
    A: tuple[float, float]
    f: tuple[float, float]
    sigma: float
    theta: tuple[float, float]

@dataclass(frozen=True)
class ParamsHat:
    alpha_ou: float
    A: tuple[float, float]
    f: tuple[float, float]
    sigma: float
    theta: tuple[float, float]
    rho: float
    lambda_: float
    alpha_mix: tuple[float, float, float]

@dataclass(frozen=True)
class NoisePreset:
    additive_sigma_scale: float
    drift_sigma_scale: float
    multiplicative_sigma_scale: float
    cumulative_increment_scale: float

@dataclass(frozen=True)
class NoiseConfig:
    preset_name: str
    seed: int
    component_mode: str
    strength_multiplier: float
    apply_scope: str

@dataclass(frozen=True)
class ReferenceSamplingConfig:
    mode: str
    sample_size: int
    num_trials: int
    seed: int
    history_window: int

@dataclass
class HistoryModalCache:
    h_source: np.ndarray
    labels_all: np.ndarray
    trial_idx: np.ndarray
    history_window: int
    ata_sum: np.ndarray
    hc_sum: np.ndarray
    h_sum: np.ndarray
    last_t: int = -1

@dataclass
class TorchHistoryModalPrecompute:
    h_time: torch.Tensor
    ata_prefix: torch.Tensor
    hc_prefix: torch.Tensor
    h_prefix: torch.Tensor
    history_window: int

NOISE_PRESETS = {
    "none": NoisePreset(
        additive_sigma_scale=0.0,
        drift_sigma_scale=0.0,
        multiplicative_sigma_scale=0.0,
        cumulative_increment_scale=0.0,
    ),
    "light": NoisePreset(
        additive_sigma_scale=0.012,
        drift_sigma_scale=0.008,
        multiplicative_sigma_scale=0.015,
        cumulative_increment_scale=0.020,
    ),
    "medium": NoisePreset(
        additive_sigma_scale=0.025,
        drift_sigma_scale=0.015,
        multiplicative_sigma_scale=0.030,
        cumulative_increment_scale=0.040,
    ),
}
NOISE_COMPONENT_MODES = (
    "none",
    "mixed",
    "additive",
    "drift",
    "multiplicative",
    "cumulative",
    "laplace",
    "impulse",
)
NOISE_APPLY_SCOPES = ("all", "test-only", "reference-only")
REFERENCE_SAMPLING_MODES = ("all", "sampled")
DEFAULT_HISTORY_WINDOW = 720
DEFAULT_PARAMS_HAT_RHO = 5.0
DEFAULT_PARAMS_HAT_LAMBDA = 0.5
DEFAULT_PARAMS_HAT_ALPHA_MIX = (0.75, 0.60, 0.15)

def parse_int_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]

def parse_float_tuple(raw: str, *, expected_len: int, option_name: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in raw.split(",") if item.strip())
    if len(values) != expected_len:
        raise ValueError(
            f"{option_name} must contain exactly {expected_len} comma-separated values; "
            f"received {len(values)}."
        )
    return values

def stable_seed(base_seed: int, user_id: int, stream_offset: int) -> int:
    return int((base_seed + 1009 * user_id + 7919 * stream_offset) % (2**32 - 1))

def smooth_gaussian_same(x: np.ndarray, window: int = 21, alpha: float = 2.5) -> np.ndarray:
    """Approximate MATLAB smoothdata(x, 'gaussian', window)."""
    if window <= 1:
        return x.astype(np.float64, copy=True)
    w = windows.gaussian(window, std=(window - 1) / (2.0 * alpha), sym=True)
    w = w / w.sum()
    pad = window // 2
    x_pad = np.pad(x.astype(np.float64), (pad, pad), mode="edge")
    out = np.convolve(x_pad, w, mode="valid")
    return out

def discretize_three_level(x: np.ndarray) -> np.ndarray:
    out = np.zeros_like(x, dtype=np.int64)
    out[(x > 2.0 / 3.0) & (x < 4.0 / 3.0)] = 1
    out[x >= 4.0 / 3.0] = 2
    return out

def confusion_matrix_3(true_labels: np.ndarray, pred_labels: np.ndarray) -> np.ndarray:
    c = np.zeros((3, 3), dtype=np.int64)
    for t, p in zip(true_labels.ravel(), pred_labels.ravel()):
        c[int(t), int(p)] += 1
    return c

def classification_metrics_from_confusion_matrix(c: np.ndarray) -> dict[str, object]:
    total = float(c.sum())
    tp = np.diag(c).astype(np.float64)
    fp = c.sum(axis=0).astype(np.float64) - tp
    fn = c.sum(axis=1).astype(np.float64) - tp
    tn = total - tp - fp - fn

    precision = tp / np.maximum(tp + fp, 1e-12)
    recall = tp / np.maximum(tp + fn, 1e-12)
    specificity = tn / np.maximum(tn + fp, 1e-12)
    f1 = 2.0 * precision * recall / np.maximum(precision + recall, 1e-12)
    support = c.sum(axis=1).astype(np.int64)

    accuracy = float(tp.sum() / max(total, 1.0))
    macro_precision = float(np.mean(precision))
    macro_recall = float(np.mean(recall))
    macro_f1 = float(np.mean(f1))
    balanced_accuracy = macro_recall
    weighted_f1 = float(np.sum(f1 * support) / max(float(support.sum()), 1.0))
    worst_class_recall = float(np.min(recall))
    worst_class_f1 = float(np.min(f1))

    return {
        "accuracy": accuracy,
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "specificity": specificity.tolist(),
        "f1": f1.tolist(),
        "support": support.tolist(),
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_accuracy,
        "weighted_f1": weighted_f1,
        "worst_class_recall": worst_class_recall,
        "worst_class_f1": worst_class_f1,
    }

def write_confusion_matrix_csv(path: Path, c: np.ndarray) -> None:
    lines = [
        "true_class,pred_0,pred_1,pred_2",
        f"class_0,{int(c[0,0])},{int(c[0,1])},{int(c[0,2])}",
        f"class_1,{int(c[1,0])},{int(c[1,1])},{int(c[1,2])}",
        f"class_2,{int(c[2,0])},{int(c[2,1])},{int(c[2,2])}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def write_user_markdown_report(path: Path, summary: dict[str, object]) -> None:
    lines = [
        f"# User {int(summary['user_id']):02d} Evaluation",
        "",
        f"- noise_preset: `{summary['noise']['preset']}`",
        f"- noise_seed: `{summary['noise']['seed']}`",
        f"- noise_component_mode: `{summary['noise']['component_mode']}`",
        f"- noise_strength_multiplier: `{summary['noise']['strength_multiplier']}`",
        f"- noise_apply_scope: `{summary['noise']['apply_scope']}`",
        f"- reference_sampling_mode: `{summary['reference_sampling']['mode']}`",
        f"- reference_sample_size: `{summary['reference_sampling']['sample_size']}`",
        f"- reference_num_trials: `{summary['reference_sampling']['num_trials']}`",
        f"- reference_sampling_seed: `{summary['reference_sampling']['seed']}`",
        f"- history_window: `{summary['reference_sampling']['history_window']}`",
        f"- runtime_seconds: `{summary['runtime_seconds']:.3f}`",
        f"- runtime_minutes: `{summary['runtime_minutes']:.3f}`",
        f"- target_days_runtime_seconds: `{summary.get('target_days_runtime_seconds', 0.0):.3f}`",
        f"- target_days_runtime_minutes: `{summary.get('target_days_runtime_minutes', 0.0):.3f}`",
        f"- accuracy_percent: `{summary['accuracy_percent']:.6f}`",
        f"- macro_f1: `{summary['macro_f1']:.6f}`",
        f"- balanced_accuracy: `{summary['balanced_accuracy']:.6f}`",
        f"- weighted_f1: `{summary['weighted_f1']:.6f}`",
        f"- worst_class_recall: `{summary['worst_class_recall']:.6f}`",
        f"- worst_class_f1: `{summary['worst_class_f1']:.6f}`",
        "",
        "## Per-Class Metrics",
        "",
        "| class | support | precision | recall | specificity | f1 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for idx in range(3):
        lines.append(
            f"| `{idx}` | `{summary['support'][idx]}` | `{summary['precision'][idx]:.6f}` | "
            f"`{summary['recall'][idx]:.6f}` | `{summary['specificity'][idx]:.6f}` | "
            f"`{summary['f1'][idx]:.6f}` |"
        )
    lines.extend(
        [
            "",
            "## Confusion Matrix",
            "",
            "| true \\ pred | 0 | 1 | 2 |",
            "| --- | --- | --- | --- |",
            f"| 0 | `{summary['confusion_matrix'][0][0]}` | `{summary['confusion_matrix'][0][1]}` | `{summary['confusion_matrix'][0][2]}` |",
            f"| 1 | `{summary['confusion_matrix'][1][0]}` | `{summary['confusion_matrix'][1][1]}` | `{summary['confusion_matrix'][1][2]}` |",
            f"| 2 | `{summary['confusion_matrix'][2][0]}` | `{summary['confusion_matrix'][2][1]}` | `{summary['confusion_matrix'][2][2]}` |",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")

def aggregate_user_summaries(
    *,
    data_dir: Path,
    out_root: Path,
    user_ids: list[int],
    summaries: list[dict[str, object]],
) -> dict[str, object]:
    metric_names = [
        "accuracy_percent",
        "macro_f1",
        "balanced_accuracy",
        "weighted_f1",
        "worst_class_recall",
        "worst_class_f1",
    ]
    pooled_confusion = np.zeros((3, 3), dtype=np.int64)
    for summary in summaries:
        pooled_confusion += np.asarray(summary["confusion_matrix"], dtype=np.int64)

    pooled_metrics = classification_metrics_from_confusion_matrix(pooled_confusion)
    aggregate_metrics: dict[str, dict[str, float]] = {}
    for metric_name in metric_names:
        values = np.asarray([float(summary[metric_name]) for summary in summaries], dtype=np.float64)
        aggregate_metrics[metric_name] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
        }

    macro_f1_values = [(int(summary["user_id"]), float(summary["macro_f1"])) for summary in summaries]
    worst_macro_user, worst_macro_value = min(macro_f1_values, key=lambda item: item[1])
    recall2_values = [(int(summary["user_id"]), float(summary["recall"][2])) for summary in summaries]
    worst_recall2_user, worst_recall2_value = min(recall2_values, key=lambda item: item[1])
    runtime_values = np.asarray([float(summary["runtime_seconds"]) for summary in summaries], dtype=np.float64)
    per_user_runtime_seconds = {
        f"user{int(summary['user_id']):02d}": float(summary["runtime_seconds"])
        for summary in summaries
    }

    return {
        "data_dir": str(data_dir),
        "out_root": str(out_root),
        "user_ids": user_ids,
        "num_users": len(user_ids),
        "noise": summaries[0]["noise"] if summaries else None,
        "reference_sampling": summaries[0]["reference_sampling"] if summaries else None,
        "runtime": {
            "total_seconds": float(runtime_values.sum()),
            "total_minutes": float(runtime_values.sum() / 60.0),
            "mean_seconds": float(runtime_values.mean()),
            "std_seconds": float(runtime_values.std()),
            "min_seconds": float(runtime_values.min()),
            "max_seconds": float(runtime_values.max()),
            "per_user_seconds": per_user_runtime_seconds,
        },
        "aggregate_metrics": aggregate_metrics,
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
        "worst_users": {
            "macro_f1": {
                "user_id": worst_macro_user,
                "value": worst_macro_value,
            },
            "recall_class_2": {
                "user_id": worst_recall2_user,
                "value": worst_recall2_value,
            },
        },
        "summaries": summaries,
    }

def write_aggregate_markdown_report(path: Path, aggregate: dict[str, object]) -> None:
    metrics = aggregate["aggregate_metrics"]
    pooled = aggregate["pooled_metrics"]
    lines = [
        "# Aggregate Evaluation",
        "",
        f"- data_dir: `{aggregate['data_dir']}`",
        f"- out_root: `{aggregate['out_root']}`",
        f"- num_users: `{aggregate['num_users']}`",
        f"- noise_preset: `{aggregate['noise']['preset']}`",
        f"- noise_seed: `{aggregate['noise']['seed']}`",
        f"- noise_component_mode: `{aggregate['noise']['component_mode']}`",
        f"- noise_strength_multiplier: `{aggregate['noise']['strength_multiplier']}`",
        f"- noise_apply_scope: `{aggregate['noise']['apply_scope']}`",
        f"- reference_sampling_mode: `{aggregate['reference_sampling']['mode']}`",
        f"- reference_sample_size: `{aggregate['reference_sampling']['sample_size']}`",
        f"- reference_num_trials: `{aggregate['reference_sampling']['num_trials']}`",
        f"- reference_sampling_seed: `{aggregate['reference_sampling']['seed']}`",
        f"- history_window: `{aggregate['reference_sampling']['history_window']}`",
        f"- runtime_total_seconds: `{aggregate['runtime']['total_seconds']:.3f}`",
        f"- runtime_total_minutes: `{aggregate['runtime']['total_minutes']:.3f}`",
        f"- runtime_mean_seconds_per_user: `{aggregate['runtime']['mean_seconds']:.3f}`",
        "",
        "## User-Level Aggregate Metrics",
        "",
        "| metric | mean | std | min | max |",
        "| --- | --- | --- | --- | --- |",
    ]
    for metric_name, metric_summary in metrics.items():
        lines.append(
            f"| `{metric_name}` | `{metric_summary['mean']:.6f}` | `{metric_summary['std']:.6f}` | "
            f"`{metric_summary['min']:.6f}` | `{metric_summary['max']:.6f}` |"
        )
    lines.extend(
        [
            "",
            "## Worst Users",
            "",
            f"- worst_macro_f1_user: `user{int(aggregate['worst_users']['macro_f1']['user_id']):02d}` "
            f"with `{aggregate['worst_users']['macro_f1']['value']:.6f}`",
            f"- worst_recall_class_2_user: `user{int(aggregate['worst_users']['recall_class_2']['user_id']):02d}` "
            f"with `{aggregate['worst_users']['recall_class_2']['value']:.6f}`",
            "",
            "## Pooled Metrics",
            "",
            f"- pooled_accuracy_percent: `{pooled['accuracy_percent']:.6f}`",
            f"- pooled_macro_f1: `{pooled['macro_f1']:.6f}`",
            f"- pooled_balanced_accuracy: `{pooled['balanced_accuracy']:.6f}`",
            f"- pooled_weighted_f1: `{pooled['weighted_f1']:.6f}`",
            "",
            "## Pooled Confusion Matrix",
            "",
            "| true \\ pred | 0 | 1 | 2 |",
            "| --- | --- | --- | --- |",
            f"| 0 | `{aggregate['pooled_confusion_matrix'][0][0]}` | `{aggregate['pooled_confusion_matrix'][0][1]}` | `{aggregate['pooled_confusion_matrix'][0][2]}` |",
            f"| 1 | `{aggregate['pooled_confusion_matrix'][1][0]}` | `{aggregate['pooled_confusion_matrix'][1][1]}` | `{aggregate['pooled_confusion_matrix'][1][2]}` |",
            f"| 2 | `{aggregate['pooled_confusion_matrix'][2][0]}` | `{aggregate['pooled_confusion_matrix'][2][1]}` | `{aggregate['pooled_confusion_matrix'][2][2]}` |",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")

def resolve_storage_dtype(name: str) -> np.dtype:
    if name == "float32":
        return np.float32
    if name == "float64":
        return np.float64
    raise ValueError(f"Unsupported storage dtype: {name}")

def resolve_torch_dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float64":
        return torch.float64
    raise ValueError(f"Unsupported torch dtype: {name}")

def resolve_torch_device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is false. "
            "Use --device cpu for local correctness smoke tests."
        )
    return device

def configure_torch_runtime() -> None:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

def load_single_variable(mat_path: Path, variable_name: str, storage_dtype: np.dtype) -> np.ndarray:
    """Load just one variable from a MAT file to avoid pulling the whole file into memory."""
    payload = loadmat(mat_path, variable_names=[variable_name])
    return np.asarray(payload[variable_name], dtype=storage_dtype)

def load_user_tensors(data_dir: Path, user_id: int, storage_dtype: np.dtype) -> tuple[list[np.ndarray], np.ndarray]:
    health_path = data_dir / "healthData.mat"
    insole_path = data_dir / "insoleData.mat"
    emr_path = data_dir / "EMRData.mat"
    h1 = load_single_variable(health_path, f"H_{user_id}_health", storage_dtype)
    h2 = load_single_variable(insole_path, f"H_{user_id}_insole", storage_dtype)
    h3 = load_single_variable(emr_path, f"H_{user_id}_EMR", storage_dtype)
    labels = load_single_variable(health_path, f"b_{user_id}", storage_dtype)
    return [h1, h2, h3], labels

def build_rowwise_drift(matrix: np.ndarray, target_scale: float, rng: np.random.Generator) -> np.ndarray:
    if target_scale <= 0.0:
        return np.zeros_like(matrix)
    steps = rng.normal(0.0, 1.0, size=matrix.shape)
    drift = np.cumsum(steps, axis=1)
    drift -= drift.mean(axis=1, keepdims=True)
    row_std = drift.std(axis=1, keepdims=True)
    row_std[row_std < 1e-12] = 1.0
    return drift / row_std * target_scale

def impulse_probability(strength: float) -> float:
    if strength <= 0.0:
        return 0.0
    return min(0.05, 0.0025 * strength)

def sample_zero_mean_noise(
    rng: np.random.Generator,
    *,
    target_std: float,
    shape: tuple[int, ...],
    distribution: str,
    impulse_prob: float = 0.0,
) -> np.ndarray:
    if target_std <= 0.0:
        return np.zeros(shape, dtype=np.float64)
    if distribution == "normal":
        return rng.normal(0.0, target_std, size=shape)
    if distribution == "laplace":
        return rng.laplace(0.0, target_std / np.sqrt(2.0), size=shape)
    if distribution == "impulse":
        if impulse_prob <= 0.0:
            return np.zeros(shape, dtype=np.float64)
        mask = rng.random(shape) < impulse_prob
        spike_std = target_std / np.sqrt(impulse_prob)
        return mask * rng.normal(0.0, spike_std, size=shape)
    raise ValueError(f"Unsupported noise distribution: {distribution}")

def apply_additive_and_drift_noise(
    matrix: np.ndarray,
    *,
    sigma_scale: float,
    drift_scale: float,
    rng: np.random.Generator,
    noise_distribution: str = "normal",
    impulse_prob: float = 0.0,
    clip_min: float | None = None,
    clip_max: float | None = None,
) -> np.ndarray:
    base_std = float(np.std(matrix))
    if base_std < 1e-12:
        return matrix.copy()
    noise = sample_zero_mean_noise(
        rng,
        target_std=sigma_scale * base_std,
        shape=matrix.shape,
        distribution=noise_distribution,
        impulse_prob=impulse_prob,
    )
    drift = build_rowwise_drift(matrix, drift_scale * base_std, rng)
    out = matrix + noise + drift
    if clip_min is not None or clip_max is not None:
        out = np.clip(
            out,
            -np.inf if clip_min is None else clip_min,
            np.inf if clip_max is None else clip_max,
        )
    return out

def apply_multiplicative_noise(
    matrix: np.ndarray,
    *,
    sigma_scale: float,
    rng: np.random.Generator,
    noise_distribution: str = "normal",
    impulse_prob: float = 0.0,
    clip_min: float | None = None,
    clip_max: float | None = None,
) -> np.ndarray:
    if sigma_scale <= 0.0:
        return matrix.copy()
    factors = 1.0 + sample_zero_mean_noise(
        rng,
        target_std=sigma_scale,
        shape=matrix.shape,
        distribution=noise_distribution,
        impulse_prob=impulse_prob,
    )
    out = matrix * factors
    if clip_min is not None or clip_max is not None:
        out = np.clip(
            out,
            -np.inf if clip_min is None else clip_min,
            np.inf if clip_max is None else clip_max,
        )
    return out

def apply_cumulative_noise(
    matrix: np.ndarray,
    *,
    increment_scale: float,
    rng: np.random.Generator,
    noise_distribution: str = "normal",
    impulse_prob: float = 0.0,
) -> np.ndarray:
    if increment_scale <= 0.0:
        return matrix.copy()
    base = matrix[:, :1]
    increments = np.diff(matrix, axis=1, prepend=base)
    inc_std = float(np.std(increments[:, 1:])) if increments.shape[1] > 1 else 0.0
    if inc_std < 1e-12:
        inc_std = max(float(np.std(matrix)), 1.0)
    noisy_increments = increments.copy()
    noisy_increments[:, 1:] = np.maximum(
        0.0,
        noisy_increments[:, 1:]
        + sample_zero_mean_noise(
            rng,
            target_std=increment_scale * inc_std,
            shape=noisy_increments[:, 1:].shape,
            distribution=noise_distribution,
            impulse_prob=impulse_prob,
        ),
    )
    out = np.empty_like(matrix)
    out[:, :1] = base
    out[:, 1:] = base + np.cumsum(noisy_increments[:, 1:], axis=1)
    return out

def selected_day_indices(scope: str) -> np.ndarray | None:
    if scope == "all":
        return None
    if scope == "test-only":
        return DAYS.astype(np.int64) - 1
    if scope == "reference-only":
        return OTHERS.astype(np.int64) - 1
    raise ValueError(f"Unsupported noise apply scope: {scope}")

def apply_noise_to_selected_days(
    matrix: np.ndarray,
    selected_idx: np.ndarray | None,
    noise_fn,
) -> np.ndarray:
    if selected_idx is None:
        return noise_fn(matrix)
    if selected_idx.size == 0:
        return matrix.copy()
    out = matrix.copy()
    out[selected_idx, :] = noise_fn(matrix[selected_idx, :])
    return out

def apply_column_aware_noise(
    h1: np.ndarray,
    h2: np.ndarray,
    *,
    config: NoiseConfig,
    user_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    preset = NOISE_PRESETS[config.preset_name]
    if config.preset_name == "none" or config.component_mode == "none":
        return h1, h2

    selected_idx = selected_day_indices(config.apply_scope)
    strength = max(float(config.strength_multiplier), 0.0)
    additive_scale = 0.0
    drift_scale = 0.0
    multiplicative_scale = 0.0
    cumulative_scale = 0.0
    noise_distribution = "normal"
    impulse_prob = 0.0
    if config.component_mode == "mixed":
        additive_scale = preset.additive_sigma_scale * strength
        drift_scale = preset.drift_sigma_scale * strength
        multiplicative_scale = preset.multiplicative_sigma_scale * strength
        cumulative_scale = preset.cumulative_increment_scale * strength
    elif config.component_mode == "additive":
        additive_scale = preset.additive_sigma_scale * strength
    elif config.component_mode == "drift":
        drift_scale = preset.drift_sigma_scale * strength
    elif config.component_mode == "multiplicative":
        multiplicative_scale = preset.multiplicative_sigma_scale * strength
    elif config.component_mode == "cumulative":
        cumulative_scale = preset.cumulative_increment_scale * strength
    elif config.component_mode == "laplace":
        additive_scale = preset.additive_sigma_scale * strength
        multiplicative_scale = preset.multiplicative_sigma_scale * strength
        cumulative_scale = preset.cumulative_increment_scale * strength
        noise_distribution = "laplace"
    elif config.component_mode == "impulse":
        additive_scale = preset.additive_sigma_scale * strength
        multiplicative_scale = preset.multiplicative_sigma_scale * strength
        cumulative_scale = preset.cumulative_increment_scale * strength
        noise_distribution = "impulse"
        impulse_prob = impulse_probability(strength)
    else:
        raise ValueError(f"Unsupported noise component mode: {config.component_mode}")

    for feature_idx, feature_name in enumerate(H1_FEATURES):
        feature_matrix = np.asarray(h1[:, feature_idx, :], dtype=np.float64)
        rng = np.random.default_rng(stable_seed(config.seed, user_id, 100 + feature_idx))

        if feature_name in H1_DISCRETE_FEATURES:
            continue
        if config.component_mode == "laplace" and feature_name not in H1_LAPLACE_FEATURES:
            continue
        if config.component_mode == "impulse" and feature_name not in H1_IMPULSE_FEATURES:
            continue
        if feature_name in H1_CUMULATIVE_FEATURES:
            h1[:, feature_idx, :] = apply_noise_to_selected_days(
                feature_matrix,
                selected_idx,
                lambda values: apply_cumulative_noise(
                    values,
                    increment_scale=cumulative_scale,
                    rng=rng,
                    noise_distribution=noise_distribution,
                    impulse_prob=impulse_prob,
                ),
            ).astype(h1.dtype, copy=False)
            continue
        if feature_name == "spo2":
            h1[:, feature_idx, :] = apply_noise_to_selected_days(
                feature_matrix,
                selected_idx,
                lambda values: apply_additive_and_drift_noise(
                    values,
                    sigma_scale=additive_scale,
                    drift_scale=drift_scale,
                    rng=rng,
                    noise_distribution=noise_distribution,
                    impulse_prob=impulse_prob,
                    clip_min=0.0,
                    clip_max=100.0,
                ),
            ).astype(h1.dtype, copy=False)
            continue
        if feature_name in H1_ADDITIVE_DRIFT_FEATURES:
            h1[:, feature_idx, :] = apply_noise_to_selected_days(
                feature_matrix,
                selected_idx,
                lambda values: apply_additive_and_drift_noise(
                    values,
                    sigma_scale=additive_scale,
                    drift_scale=drift_scale,
                    rng=rng,
                    noise_distribution=noise_distribution,
                    impulse_prob=impulse_prob,
                    clip_min=0.0,
                ),
            ).astype(h1.dtype, copy=False)
            continue

    for feature_idx, feature_name in enumerate(H2_FEATURES):
        feature_matrix = np.asarray(h2[:, feature_idx, :], dtype=np.float64)
        rng = np.random.default_rng(stable_seed(config.seed, user_id, 200 + feature_idx))

        if feature_name in H2_DISCRETE_FEATURES:
            continue
        if config.component_mode == "laplace" and feature_name not in H2_LAPLACE_FEATURES:
            continue
        if config.component_mode == "impulse" and feature_name not in H2_IMPULSE_FEATURES:
            continue
        if feature_name in H2_MULTIPLICATIVE_FEATURES:
            h2[:, feature_idx, :] = apply_noise_to_selected_days(
                feature_matrix,
                selected_idx,
                lambda values: apply_multiplicative_noise(
                    values,
                    sigma_scale=multiplicative_scale,
                    rng=rng,
                    noise_distribution=noise_distribution,
                    impulse_prob=impulse_prob,
                    clip_min=0.0,
                ),
            ).astype(h2.dtype, copy=False)
            continue
        if feature_name in H2_RATIO_FEATURES:
            clip_max = 1.0 if feature_name == "lr_ratio" else 100.0
            h2[:, feature_idx, :] = apply_noise_to_selected_days(
                feature_matrix,
                selected_idx,
                lambda values: apply_additive_and_drift_noise(
                    values,
                    sigma_scale=additive_scale,
                    drift_scale=drift_scale,
                    rng=rng,
                    noise_distribution=noise_distribution,
                    impulse_prob=impulse_prob,
                    clip_min=0.0,
                    clip_max=clip_max,
                ),
            ).astype(h2.dtype, copy=False)
            continue
        if feature_name in H2_CENTER_FEATURES:
            h2[:, feature_idx, :] = apply_noise_to_selected_days(
                feature_matrix,
                selected_idx,
                lambda values: apply_additive_and_drift_noise(
                    values,
                    sigma_scale=additive_scale,
                    drift_scale=drift_scale,
                    rng=rng,
                    noise_distribution=noise_distribution,
                    impulse_prob=impulse_prob,
                ),
            ).astype(h2.dtype, copy=False)
            continue

    return h1, h2

def validate_reference_sampling_config(
    config: ReferenceSamplingConfig,
    reference_pool_size: int,
) -> None:
    if config.mode not in REFERENCE_SAMPLING_MODES:
        raise ValueError(f"Unsupported reference sampling mode: {config.mode}")
    if config.sample_size <= 0:
        raise ValueError("--reference-sample-size must be positive.")
    if config.num_trials <= 0:
        raise ValueError("--reference-num-trials must be positive.")
    if config.history_window <= 0:
        raise ValueError("--history-window must be positive.")
    if config.mode == "sampled" and config.sample_size > reference_pool_size:
        raise ValueError(
            f"--reference-sample-size={config.sample_size} exceeds reference pool size "
            f"{reference_pool_size}."
        )

def effective_reference_sampling_config(
    config: ReferenceSamplingConfig,
    reference_pool_size: int,
) -> ReferenceSamplingConfig:
    validate_reference_sampling_config(config, reference_pool_size=reference_pool_size)
    if config.mode == "all":
        return ReferenceSamplingConfig(
            mode=config.mode,
            sample_size=reference_pool_size,
            num_trials=1,
            seed=config.seed,
            history_window=config.history_window,
        )
    return config

def create_history_modal_cache(
    *,
    h_source: np.ndarray,
    labels_all: np.ndarray,
    trial_idx: np.ndarray,
    history_window: int,
) -> HistoryModalCache:
    feature_dim = int(h_source.shape[1])
    return HistoryModalCache(
        h_source=h_source,
        labels_all=labels_all,
        trial_idx=np.asarray(trial_idx, dtype=np.int64),
        history_window=int(history_window),
        ata_sum=np.zeros((feature_dim, feature_dim), dtype=np.float64),
        hc_sum=np.zeros(feature_dim, dtype=np.float64),
        h_sum=np.zeros(feature_dim, dtype=np.float64),
    )

def advance_history_modal_cache(
    cache: HistoryModalCache,
    target_t: int,
) -> None:
    if target_t < cache.last_t:
        raise ValueError("History cache cannot move backwards.")
    if target_t == cache.last_t:
        return

    trial_idx = cache.trial_idx
    for t in range(cache.last_t + 1, target_t + 1):
        hsamp = np.asarray(cache.h_source[trial_idx, :, t], dtype=np.float64)
        csamp = np.asarray(cache.labels_all[trial_idx, t], dtype=np.float64)
        cache.ata_sum += hsamp.T @ hsamp
        cache.hc_sum += hsamp.T @ csamp
        cache.h_sum += hsamp.sum(axis=0)

        remove_t = t - cache.history_window
        if remove_t >= 0:
            hsamp_old = np.asarray(cache.h_source[trial_idx, :, remove_t], dtype=np.float64)
            csamp_old = np.asarray(cache.labels_all[trial_idx, remove_t], dtype=np.float64)
            cache.ata_sum -= hsamp_old.T @ hsamp_old
            cache.hc_sum -= hsamp_old.T @ csamp_old
            cache.h_sum -= hsamp_old.sum(axis=0)

    cache.last_t = target_t

def solve_history_modal_y(
    *,
    b: np.ndarray,
    cache: HistoryModalCache,
    current_t: int,
    xhat_t: float,
    params_hat: ParamsHat,
) -> np.ndarray:
    advance_history_modal_cache(cache, target_t=current_t)
    window_start = max(0, current_t - cache.history_window + 1)
    window_count = current_t - window_start + 1
    if window_count <= 0:
        raise ValueError("History window is empty.")

    # H^T H, H^T c, and H^T 1 are rolling sums; only H^T w depends on current B.
    weighted_h_sum = np.zeros_like(cache.h_sum)
    chunk_size = 128
    trial_idx = cache.trial_idx
    b_view = np.asarray(b, dtype=np.float64)
    for chunk_start in range(window_start, current_t + 1, chunk_size):
        chunk_end = min(current_t + 1, chunk_start + chunk_size)
        hist = np.asarray(cache.h_source[trial_idx, :, chunk_start:chunk_end], dtype=np.float64)
        hist_r = np.moveaxis(hist, 2, 0)
        diff = hist_r - b_view[None, None, :]
        dist = np.sum(diff * diff, axis=2)
        h_val = np.maximum(np.median(dist, axis=1), 1e-12)
        scores = -dist / np.square(h_val)[:, None]
        scores -= np.max(scores, axis=1, keepdims=True)
        weights = np.exp(scores)
        weight_sum = np.sum(weights, axis=1, keepdims=True)
        uniform = 1.0 / float(weights.shape[1])
        weights = np.divide(
            weights,
            weight_sum,
            out=np.full_like(weights, uniform, dtype=np.float64),
            where=weight_sum > 1e-12,
        )
        weighted_h_sum += np.einsum("rsm,rs->m", hist_r, weights)

    ata = params_hat.rho * np.outer(b_view, b_view) + cache.ata_sum / float(window_count)
    atb = (
        params_hat.rho * xhat_t * b_view
        + (
            cache.hc_sum
            - params_hat.lambda_ * xhat_t * (cache.h_sum - weighted_h_sum)
        )
        / float(window_count)
    )
    try:
        return np.linalg.solve(ata, atb)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(ata) @ atb

def solve_history_modal_y_batch(
    *,
    b: np.ndarray,
    caches: list[HistoryModalCache],
    current_t: int,
    xhat_t: float,
    params_hat: ParamsHat,
) -> np.ndarray:
    if not caches:
        raise ValueError("History cache batch is empty.")

    for cache in caches:
        advance_history_modal_cache(cache, target_t=current_t)

    history_window = caches[0].history_window
    window_start = max(0, current_t - history_window + 1)
    window_count = current_t - window_start + 1
    if window_count <= 0:
        raise ValueError("History window is empty.")

    h_source = caches[0].h_source
    trial_idx = np.stack([cache.trial_idx for cache in caches], axis=0)
    b_view = np.asarray(b, dtype=np.float64)
    weighted_h_sum = np.zeros((len(caches), h_source.shape[1]), dtype=np.float64)
    chunk_size = 128
    for chunk_start in range(window_start, current_t + 1, chunk_size):
        chunk_end = min(current_t + 1, chunk_start + chunk_size)
        hist = np.asarray(h_source[trial_idx, :, chunk_start:chunk_end], dtype=np.float64)
        hist_r = np.moveaxis(hist, 3, 0)
        diff = hist_r - b_view[None, None, None, :]
        dist = np.sum(diff * diff, axis=3)
        h_val = np.maximum(np.median(dist, axis=2), 1e-12)
        scores = -dist / np.square(h_val)[:, :, None]
        scores -= np.max(scores, axis=2, keepdims=True)
        weights = np.exp(scores)
        weight_sum = np.sum(weights, axis=2, keepdims=True)
        uniform = 1.0 / float(weights.shape[2])
        weights = np.divide(
            weights,
            weight_sum,
            out=np.full_like(weights, uniform, dtype=np.float64),
            where=weight_sum > 1e-12,
        )
        weighted_h_sum += np.einsum("rskm,rsk->sm", hist_r, weights)

    ata_sum = np.stack([cache.ata_sum for cache in caches], axis=0)
    hc_sum = np.stack([cache.hc_sum for cache in caches], axis=0)
    h_sum = np.stack([cache.h_sum for cache in caches], axis=0)
    ata = params_hat.rho * np.outer(b_view, b_view)[None, :, :] + ata_sum / float(window_count)
    atb = (
        params_hat.rho * xhat_t * b_view[None, :]
        + (hc_sum - params_hat.lambda_ * xhat_t * (h_sum - weighted_h_sum))
        / float(window_count)
    )
    try:
        return np.linalg.solve(ata, atb[..., None])[..., 0]
    except np.linalg.LinAlgError:
        return np.stack([np.linalg.pinv(a) @ rhs for a, rhs in zip(ata, atb)], axis=0)

def precompute_static_modal_terms(
    *,
    b: np.ndarray,
    hsamp: np.ndarray,
    csamp: np.ndarray,
    params_hat: ParamsHat,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dist = np.sum((hsamp - b) ** 2, axis=1)
    h_val = max(float(np.median(dist)), 1e-12)
    w = np.exp(-dist / (h_val * h_val))
    w_sum = float(w.sum())
    if w_sum <= 1e-12:
        w = np.full_like(w, 1.0 / max(len(w), 1), dtype=np.float64)
    else:
        w = w / w_sum
    n = params_hat.lambda_ * (1.0 - w)
    return hsamp.T @ hsamp, hsamp.T @ csamp, hsamp.T @ n

def solve_static_modal_y(
    *,
    b: np.ndarray,
    h_th: np.ndarray,
    h_tc: np.ndarray,
    h_tn: np.ndarray,
    xhat_t: float,
    params_hat: ParamsHat,
) -> np.ndarray:
    ata = params_hat.rho * np.outer(b, b) + h_th
    atb = params_hat.rho * xhat_t * b + h_tc - h_tn * xhat_t
    try:
        return np.linalg.solve(ata, atb)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(ata) @ atb

def solve_static_modal_y_batch(
    *,
    b: np.ndarray,
    h_th: np.ndarray,
    h_tc: np.ndarray,
    h_tn: np.ndarray,
    xhat_t: float,
    params_hat: ParamsHat,
) -> np.ndarray:
    b_view = np.asarray(b, dtype=np.float64)
    ata = params_hat.rho * np.outer(b_view, b_view)[None, :, :] + np.asarray(
        h_th, dtype=np.float64
    )
    atb = (
        params_hat.rho * xhat_t * b_view[None, :]
        + np.asarray(h_tc, dtype=np.float64)
        - np.asarray(h_tn, dtype=np.float64) * xhat_t
    )
    try:
        return np.linalg.solve(ata, atb[..., None])[..., 0]
    except np.linalg.LinAlgError:
        return np.stack([np.linalg.pinv(a) @ rhs for a, rhs in zip(ata, atb)], axis=0)

def build_reference_trial_indices(
    *,
    sample_idx: np.ndarray,
    config: ReferenceSamplingConfig,
    user_id: int,
) -> list[np.ndarray]:
    config = effective_reference_sampling_config(config, reference_pool_size=int(sample_idx.size))
    if config.mode == "all":
        return [sample_idx]

    rng = np.random.default_rng(stable_seed(config.seed, user_id, 50000))
    return [
        rng.choice(sample_idx, size=config.sample_size, replace=False).astype(np.int64, copy=False)
        for _ in range(config.num_trials)
    ]

def build_torch_history_modal_precompute(
    *,
    h_source: torch.Tensor,
    labels_all: torch.Tensor,
    trial_idx: torch.Tensor,
    history_window: int,
) -> TorchHistoryModalPrecompute:
    h_selected = h_source[trial_idx]
    labels_selected = labels_all[trial_idx]
    h_time = h_selected.permute(0, 3, 1, 2).contiguous()
    labels_time = labels_selected.permute(0, 2, 1).contiguous()
    ata_prefix = torch.cumsum(torch.einsum("stkm,stkn->stmn", h_time, h_time), dim=1)
    hc_prefix = torch.cumsum(torch.einsum("stkm,stk->stm", h_time, labels_time), dim=1)
    h_prefix = torch.cumsum(h_time.sum(dim=2), dim=1)
    return TorchHistoryModalPrecompute(
        h_time=h_time,
        ata_prefix=ata_prefix,
        hc_prefix=hc_prefix,
        h_prefix=h_prefix,
        history_window=int(history_window),
    )

def _torch_prefix_window(prefix: torch.Tensor, window_start: int, current_t: int) -> torch.Tensor:
    total = prefix[:, current_t]
    if window_start <= 0:
        return total
    return total - prefix[:, window_start - 1]

def _torch_numpy_median(values: torch.Tensor, dim: int) -> torch.Tensor:
    count = values.shape[dim]
    if count % 2 == 1:
        return torch.median(values, dim=dim).values
    lower = torch.kthvalue(values, count // 2, dim=dim).values
    upper = torch.kthvalue(values, count // 2 + 1, dim=dim).values
    return 0.5 * (lower + upper)

def solve_batched_torch(ata: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    solution, info = torch.linalg.solve_ex(ata, rhs, check_errors=False)
    bad_mask = info != 0
    bad_count = int(torch.count_nonzero(bad_mask).detach().cpu())
    if bad_count <= 0:
        return solution

    solution = solution.clone()
    solution[bad_mask] = torch.matmul(torch.linalg.pinv(ata[bad_mask]), rhs[bad_mask])
    return solution

def solve_history_modal_y_days_torch(
    *,
    b: torch.Tensor,
    xhat_t: torch.Tensor,
    cache: TorchHistoryModalPrecompute,
    current_t: int,
    params_hat: ParamsHat,
    history_chunk_size: int,
) -> torch.Tensor:
    window_start = max(0, current_t - cache.history_window + 1)
    window_count = current_t - window_start + 1
    if window_count <= 0:
        raise ValueError("History window is empty.")

    b_view = b.to(dtype=cache.h_time.dtype)
    num_days = b_view.shape[0]
    num_trials = cache.h_time.shape[0]
    feature_dim = cache.h_time.shape[3]
    weighted_h_sum = torch.zeros(
        (num_days, num_trials, feature_dim),
        device=cache.h_time.device,
        dtype=cache.h_time.dtype,
    )

    chunk_size = max(1, int(history_chunk_size))
    for chunk_start in range(window_start, current_t + 1, chunk_size):
        chunk_end = min(current_t + 1, chunk_start + chunk_size)
        hist = cache.h_time[:, chunk_start:chunk_end, :, :]
        diff = hist[None, :, :, :, :] - b_view[:, None, None, None, :]
        dist = torch.sum(diff * diff, dim=4)
        h_val = torch.clamp(_torch_numpy_median(dist, dim=3), min=1e-12)
        scores = -dist / torch.square(h_val[:, :, :, None])
        scores = scores - torch.max(scores, dim=3, keepdim=True).values
        weights = torch.exp(scores)
        weight_sum = torch.sum(weights, dim=3, keepdim=True)
        weights = torch.where(
            weight_sum > 1e-12,
            weights / weight_sum,
            torch.full_like(weights, 1.0 / float(weights.shape[3])),
        )
        weighted_h_sum += torch.einsum("sckm,dsck->dsm", hist, weights)

    ata_sum = _torch_prefix_window(cache.ata_prefix, window_start, current_t)
    hc_sum = _torch_prefix_window(cache.hc_prefix, window_start, current_t)
    h_sum = _torch_prefix_window(cache.h_prefix, window_start, current_t)
    ata = (
        params_hat.rho * torch.einsum("dm,dn->dmn", b_view, b_view)[:, None, :, :]
        + ata_sum[None, :, :, :] / float(window_count)
    )
    atb = (
        params_hat.rho * xhat_t[:, None, None] * b_view[:, None, :]
        + (hc_sum[None, :, :] - params_hat.lambda_ * xhat_t[:, None, None] * (h_sum[None, :, :] - weighted_h_sum))
        / float(window_count)
    )
    solution = solve_batched_torch(
        ata.reshape(num_days * num_trials, feature_dim, feature_dim),
        atb.reshape(num_days * num_trials, feature_dim, 1),
    )
    return solution.reshape(num_days, num_trials, feature_dim)

def precompute_static_modal_terms_days_torch(
    *,
    b: torch.Tensor,
    hsamp: torch.Tensor,
    csamp: torch.Tensor,
    params_hat: ParamsHat,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    b_view = b.to(dtype=hsamp.dtype)
    num_days = b_view.shape[0]
    h_th = torch.einsum("skm,skn->smn", hsamp, hsamp)
    h_tc = torch.einsum("skm,sk->sm", hsamp, csamp)
    dist = torch.sum(torch.square(hsamp[None, :, :, :] - b_view[:, None, None, :]), dim=3)
    h_val = torch.clamp(_torch_numpy_median(dist, dim=2), min=1e-12)
    w = torch.exp(-dist / torch.square(h_val[:, :, None]))
    w_sum = torch.sum(w, dim=2, keepdim=True)
    w = torch.where(
        w_sum > 1e-12,
        w / w_sum,
        torch.full_like(w, 1.0 / float(w.shape[2])),
    )
    n = params_hat.lambda_ * (1.0 - w)
    h_tn = torch.einsum("skm,dsk->dsm", hsamp, n)
    return (
        h_th[None, :, :, :].expand(num_days, -1, -1, -1).contiguous(),
        h_tc[None, :, :].expand(num_days, -1, -1).contiguous(),
        h_tn,
    )

def solve_static_modal_y_days_torch(
    *,
    b: torch.Tensor,
    h_th: torch.Tensor,
    h_tc: torch.Tensor,
    h_tn: torch.Tensor,
    xhat_t: torch.Tensor,
    params_hat: ParamsHat,
) -> torch.Tensor:
    b_view = b.to(dtype=h_th.dtype)
    num_days, num_trials, feature_dim = h_tc.shape
    ata = params_hat.rho * torch.einsum("dm,dn->dmn", b_view, b_view)[:, None, :, :] + h_th
    atb = params_hat.rho * xhat_t[:, None, None] * b_view[:, None, :] + h_tc - h_tn * xhat_t[:, None, None]
    solution = solve_batched_torch(
        ata.reshape(num_days * num_trials, feature_dim, feature_dim),
        atb.reshape(num_days * num_trials, feature_dim, 1),
    )
    return solution.reshape(num_days, num_trials, feature_dim)

def run_day_batch_equivalent_torch(
    *,
    day_indices_1based: list[int],
    user_id: int,
    h_all: list[torch.Tensor],
    labels_all: torch.Tensor,
    reference_trial_idx: torch.Tensor,
    history_precomputes: list[TorchHistoryModalPrecompute],
    params_true: ParamsTrue,
    params_hat: ParamsHat,
    t_day: int,
    x_bounds: tuple[float, float],
    progress_interval: int,
    history_chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, float]:
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
    h3_th, h3_tc, h3_tn = precompute_static_modal_terms_days_torch(
        b=b3,
        hsamp=hsamp3,
        csamp=csamp3,
        params_hat=params_hat,
    )

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
        y1_trials = solve_history_modal_y_days_torch(
            b=b1,
            xhat_t=xhat[:, t],
            cache=history_precomputes[0],
            current_t=t,
            params_hat=params_hat,
            history_chunk_size=history_chunk_size,
        )
        y2_trials = solve_history_modal_y_days_torch(
            b=b2,
            xhat_t=xhat[:, t],
            cache=history_precomputes[1],
            current_t=t,
            params_hat=params_hat,
            history_chunk_size=history_chunk_size,
        )
        y3_trials = solve_static_modal_y_days_torch(
            b=b3,
            h_th=h3_th,
            h_tc=h3_tc,
            h_tn=h3_tn,
            xhat_t=xhat[:, t],
            params_hat=params_hat,
        )
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
                f"[user {user_id}] day_batch={day_indices_1based} "
                f"t={t + 1}/{t_day} ({pct:.1f}%) "
                f"xhat_mean={float(torch.mean(xhat[:, t + 1]).detach().cpu()):.6f} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    runtime_seconds = time.time() - started
    x_np = x[:, 1:].detach().cpu().numpy()
    xhat_np = xhat[:, 1:].detach().cpu().numpy()
    x_sm = np.stack([smooth_gaussian_same(row, window=21) for row in x_np], axis=0)
    xhat_sm = np.stack([smooth_gaussian_same(row, window=21) for row in xhat_np], axis=0)
    return x_sm, xhat_sm, runtime_seconds

def run_single_day_equivalent(
    day_idx_1based: int,
    user_id: int,
    h_all: list[np.ndarray],
    labels_all: np.ndarray,
    reference_trial_indices: list[np.ndarray],
    params_true: ParamsTrue,
    params_hat: ParamsHat,
    t_day: int,
    x_bounds: tuple[float, float],
    reference_sampling_config: ReferenceSamplingConfig,
    progress_interval: int,
) -> tuple[np.ndarray, np.ndarray]:
    day_idx = int(day_idx_1based) - 1
    b_day = np.asarray(labels_all[day_idx, :], dtype=np.float64)
    x = np.zeros(t_day + 1, dtype=np.float64)
    xhat = np.zeros(t_day + 1, dtype=np.float64)
    xhat[0] = 0.5
    lb, ub = x_bounds

    b3 = np.asarray(h_all[2][day_idx, :], dtype=np.float64)
    trial_indices = reference_trial_indices
    history_caches = [
        [
            create_history_modal_cache(
                h_source=h_all[i],
                labels_all=labels_all,
                trial_idx=trial_idx,
                history_window=reference_sampling_config.history_window,
            )
            for i in range(2)
        ]
        for trial_idx in trial_indices
    ]
    modal_history_caches = [
        [trial_caches[i] for trial_caches in history_caches]
        for i in range(2)
    ]
    h3_terms = []
    for trial_idx in trial_indices:
        hsamp3 = np.asarray(h_all[2][trial_idx, :], dtype=np.float64)
        csamp3 = np.asarray(labels_all[trial_idx, :], dtype=np.float64).mean(axis=1)
        h3_terms.append(
            precompute_static_modal_terms(
                b=b3,
                hsamp=hsamp3,
                csamp=csamp3,
                params_hat=params_hat,
            )
        )
    h3_th = np.stack([term[0] for term in h3_terms], axis=0)
    h3_tc = np.stack([term[1] for term in h3_terms], axis=0)
    h3_tn = np.stack([term[2] for term in h3_terms], axis=0)

    day_started = time.time()
    for t in range(t_day):
        m_t = b_day[t]
        s_true = 0.5 * (1.0 + np.tanh((m_t - np.asarray(params_true.theta)) / params_true.sigma))
        ell_true = float(np.sum(s_true))
        q_true = (
            params_true.A[0] * np.sin(2.0 * np.pi * params_true.f[0] * t)
            + params_true.A[1] * np.sin(2.0 * np.pi * params_true.f[1] * t)
        )
        x_temp = params_true.alpha_ou * x[t] + (1.0 - params_true.alpha_ou) * (ell_true + q_true)
        x[t + 1] = min(max(x_temp, lb), ub)

        b1 = np.asarray(h_all[0][day_idx, :, t], dtype=np.float64)
        b2 = np.asarray(h_all[1][day_idx, :, t], dtype=np.float64)
        bs = [b1, b2, b3]
        y1_trials = solve_history_modal_y_batch(
            b=bs[0],
            caches=modal_history_caches[0],
            current_t=t,
            xhat_t=float(xhat[t]),
            params_hat=params_hat,
        )
        y2_trials = solve_history_modal_y_batch(
            b=bs[1],
            caches=modal_history_caches[1],
            current_t=t,
            xhat_t=float(xhat[t]),
            params_hat=params_hat,
        )
        y3_trials = solve_static_modal_y_batch(
            b=b3,
            h_th=h3_th,
            h_tc=h3_tc,
            h_tn=h3_tn,
            xhat_t=float(xhat[t]),
            params_hat=params_hat,
        )
        mhat_trials = (
            params_hat.alpha_mix[0] * (y1_trials @ bs[0])
            + params_hat.alpha_mix[1] * (y2_trials @ bs[1])
            + params_hat.alpha_mix[2] * (y3_trials @ bs[2])
        )
        mhat = float(np.mean(mhat_trials))
        ell_hat = mhat
        q_hat = (
            params_hat.A[0] * np.sin(2.0 * np.pi * params_hat.f[0] * t)
            + params_hat.A[1] * np.sin(2.0 * np.pi * params_hat.f[1] * t)
        )
        xhat_temp = params_hat.alpha_ou * xhat[t] + (1.0 - params_hat.alpha_ou) * (ell_hat + q_hat)
        xhat[t + 1] = min(max(xhat_temp, lb), ub)
        if progress_interval > 0 and (
            t == 0 or (t + 1) % progress_interval == 0 or (t + 1) == t_day
        ):
            elapsed = time.time() - day_started
            pct = 100.0 * float(t + 1) / float(t_day)
            print(
                f"[user {user_id}] day_idx={day_idx_1based} "
                f"t={t + 1}/{t_day} ({pct:.1f}%) "
                f"xhat={xhat[t + 1]:.6f} elapsed={elapsed:.1f}s",
                flush=True,
            )

    x_sm = smooth_gaussian_same(x[1:], window=21)
    xhat_sm = smooth_gaussian_same(xhat[1:], window=21)
    return x_sm, xhat_sm

def run_user(
    data_dir: Path,
    user_id: int,
    out_dir: Path,
    noise_config: NoiseConfig,
    reference_sampling_config: ReferenceSamplingConfig,
    storage_dtype: np.dtype,
    progress_interval: int,
    device: torch.device,
    torch_dtype: torch.dtype,
    target_day_batch_size: int,
    history_chunk_size: int,
    verify_against_cpu: bool,
    params_hat_rho: float,
    params_hat_lambda: float,
    params_hat_alpha_mix: tuple[float, float, float],
) -> dict:
    user_started = time.time()
    reference_sampling_config = effective_reference_sampling_config(
        reference_sampling_config,
        reference_pool_size=len(OTHERS),
    )
    params_true = ParamsTrue(
        alpha_ou=float(np.exp(-4.0)),
        A=(0.15, 0.17),
        f=(0.025, 0.021),
        sigma=0.5,
        theta=(0.5, 1.5),
    )
    params_hat = ParamsHat(
        alpha_ou=float(np.exp(-4.0)),
        A=(0.11, 0.13),
        f=(0.035, 0.038),
        sigma=0.4,
        theta=(0.5, 1.5),
        rho=float(params_hat_rho),
        lambda_=float(params_hat_lambda),
        alpha_mix=tuple(float(value) for value in params_hat_alpha_mix),
    )
    h_all, labels_all = load_user_tensors(data_dir, user_id, storage_dtype)
    h_all[0], h_all[1] = apply_column_aware_noise(
        h_all[0],
        h_all[1],
        config=noise_config,
        user_id=user_id,
    )
    t_day = int(labels_all.shape[1])
    x_all = np.zeros((len(DAYS), t_day), dtype=np.float64)
    xhat_all = np.zeros((len(DAYS), t_day), dtype=np.float64)
    sample_idx = OTHERS.astype(np.int64) - 1
    reference_trial_indices = build_reference_trial_indices(
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
        build_torch_history_modal_precompute(
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
    day_summaries: list[dict[str, object]] = []

    days_list = DAYS.tolist()
    batch_size = max(1, int(target_day_batch_size))
    target_days_started = time.time()
    for batch_start in range(0, len(days_list), batch_size):
        batch_days = days_list[batch_start : batch_start + batch_size]
        print(
            f"[user {user_id}] day batch "
            f"{batch_start + 1}-{batch_start + len(batch_days)}/{len(days_list)} "
            f"-> day_idx={batch_days}",
            flush=True,
        )
        x_batch, xhat_batch, batch_runtime_seconds = run_day_batch_equivalent_torch(
            day_indices_1based=batch_days,
            user_id=user_id,
            h_all=h_all_torch,
            labels_all=labels_all_torch,
            reference_trial_idx=reference_trial_idx_torch,
            history_precomputes=history_precomputes,
            params_true=params_true,
            params_hat=params_hat,
            t_day=t_day,
            x_bounds=(0.0, 2.0),
            progress_interval=progress_interval,
            history_chunk_size=history_chunk_size,
        )
        print(
            f"[user {user_id}] day batch "
            f"{batch_start + 1}-{batch_start + len(batch_days)}/{len(days_list)} "
            f"runtime={batch_runtime_seconds:.3f}s",
            flush=True,
        )
        for batch_offset, day in enumerate(batch_days):
            idx = batch_start + batch_offset
            x_sm = x_batch[batch_offset]
            xhat_sm = xhat_batch[batch_offset]
            if verify_against_cpu:
                cpu_x, cpu_xhat = run_single_day_equivalent(
                    day_idx_1based=day,
                    user_id=user_id,
                    h_all=h_all,
                    labels_all=labels_all,
                    reference_trial_indices=reference_trial_indices,
                    params_true=params_true,
                    params_hat=params_hat,
                    t_day=t_day,
                    x_bounds=(0.0, 2.0),
                    reference_sampling_config=reference_sampling_config,
                    progress_interval=0,
                )
                cpu_pred = discretize_three_level(cpu_xhat.reshape(1, -1))
                gpu_pred = discretize_three_level(xhat_sm.reshape(1, -1))
                if not np.array_equal(cpu_pred, gpu_pred):
                    max_diff = float(np.max(np.abs(cpu_xhat - xhat_sm)))
                    raise AssertionError(
                        f"GPU/CPU classification mismatch for user {user_id}, day {day}; "
                        f"max_xhat_diff={max_diff:.12g}"
                    )
            x_all[idx, :] = x_sm
            xhat_all[idx, :] = xhat_sm
            day_true_disc = discretize_three_level(x_sm.reshape(1, -1))
            day_pred_disc = discretize_three_level(xhat_sm.reshape(1, -1))
            day_c = confusion_matrix_3(day_true_disc, day_pred_disc)
            day_metrics = classification_metrics_from_confusion_matrix(day_c)
            day_stem = f"day{idx + 1:02d}_idx{int(day):03d}"
            day_npz_path = day_out_dir / f"{day_stem}_outputs.npz"
            day_json_path = day_out_dir / f"{day_stem}_summary.json"
            np.savez_compressed(
                day_npz_path,
                x=x_sm,
                xhat=xhat_sm,
                true_disc=day_true_disc[0],
                pred_disc=day_pred_disc[0],
                day_1based=int(day),
            )
            day_summary = {
                "user_id": user_id,
                "day_order": idx + 1,
                "num_days": len(DAYS),
                "day_1based": int(day),
                "t_day": t_day,
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
            day_json_path.write_text(json.dumps(day_summary, indent=2), encoding="utf-8")
            write_confusion_matrix_csv(day_out_dir / f"{day_stem}_confusion_matrix.csv", day_c)
            day_summaries.append(day_summary)
            (out_dir / "partial_day_summaries.json").write_text(
                json.dumps(
                    {
                        "user_id": user_id,
                        "completed_days": len(day_summaries),
                        "num_days": len(DAYS),
                        "day_summaries": day_summaries,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(
                f"[user {user_id}] day {idx + 1}/{len(DAYS)} result -> "
                f"accuracy={day_summary['accuracy_percent']:.4f}% "
                f"macro_f1={day_summary['macro_f1']:.6f} "
                f"balanced_acc={day_summary['balanced_accuracy']:.6f} "
                f"xhat_range=[{day_summary['xhat_min']:.6f}, {day_summary['xhat_max']:.6f}] "
                f"xhat_final={day_summary['xhat_final']:.6f}",
                flush=True,
            )
            print(f"[user {user_id}] day {idx + 1}/{len(DAYS)} confusion={day_c.tolist()}", flush=True)
            print(f"[user {user_id}] day {idx + 1}/{len(DAYS)} saved -> {day_json_path}", flush=True)

    target_days_runtime_seconds = time.time() - target_days_started
    print(
        f"[user {user_id}] target_days_runtime={target_days_runtime_seconds:.3f}s "
        f"({target_days_runtime_seconds / 60.0:.3f} min) "
        f"for {len(DAYS)} days",
        flush=True,
    )

    true_disc = discretize_three_level(x_all)
    pred_disc = discretize_three_level(xhat_all)
    c = confusion_matrix_3(true_disc, pred_disc)
    metrics = classification_metrics_from_confusion_matrix(c)

    np.savez_compressed(
        out_dir / "user_run_outputs.npz",
        x_all=x_all,
        xhat_all=xhat_all,
        true_disc=true_disc,
        pred_disc=pred_disc,
        days=DAYS,
    )
    runtime_seconds = time.time() - user_started
    summary = {
        "user_id": user_id,
        "days_1based": DAYS.tolist(),
        "sample_days_1based": OTHERS.tolist(),
        "reference_sampling": {
            "mode": reference_sampling_config.mode,
            "sample_size": reference_sampling_config.sample_size,
            "num_trials": reference_sampling_config.num_trials,
            "seed": reference_sampling_config.seed,
            "history_window": reference_sampling_config.history_window,
        },
        "t_day": t_day,
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
            "preset": noise_config.preset_name,
            "seed": noise_config.seed,
            "component_mode": noise_config.component_mode,
            "strength_multiplier": noise_config.strength_multiplier,
            "apply_scope": noise_config.apply_scope,
        },
        "compute": {
            "backend": "torch_exact",
            "device": str(device),
            "torch_dtype": str(torch_dtype).replace("torch.", ""),
            "target_day_batch_size": target_day_batch_size,
            "history_chunk_size": history_chunk_size,
            "verify_against_cpu": verify_against_cpu,
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
    write_confusion_matrix_csv(out_dir / "confusion_matrix.csv", c)
    write_user_markdown_report(out_dir / "report.md", summary)
    print(
        f"[user {user_id}] runtime={runtime_seconds:.3f}s "
        f"({runtime_seconds / 60.0:.3f} min)",
        flush=True,
    )
    del h_all, labels_all, h_all_torch, labels_all_torch, x_all, xhat_all, true_disc, pred_disc, c
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary

def run_users(
    data_dir: Path,
    user_ids: list[int],
    out_root: Path,
    noise_config: NoiseConfig,
    reference_sampling_config: ReferenceSamplingConfig,
    storage_dtype: np.dtype,
    progress_interval: int,
    device: torch.device,
    torch_dtype: torch.dtype,
    target_day_batch_size: int,
    history_chunk_size: int,
    verify_against_cpu: bool,
    params_hat_rho: float,
    params_hat_lambda: float,
    params_hat_alpha_mix: tuple[float, float, float],
) -> list[dict]:
    summaries: list[dict] = []
    for user_id in user_ids:
        user_out_dir = out_root / f"user{user_id:02d}"
        print(f"=== Running user {user_id} -> {user_out_dir} ===", flush=True)
        summary = run_user(
            data_dir=data_dir,
            user_id=user_id,
            out_dir=user_out_dir,
            noise_config=noise_config,
            reference_sampling_config=reference_sampling_config,
            storage_dtype=storage_dtype,
            progress_interval=progress_interval,
            device=device,
            torch_dtype=torch_dtype,
            target_day_batch_size=target_day_batch_size,
            history_chunk_size=history_chunk_size,
            verify_against_cpu=verify_against_cpu,
            params_hat_rho=params_hat_rho,
            params_hat_lambda=params_hat_lambda,
            params_hat_alpha_mix=params_hat_alpha_mix,
        )
        summaries.append(summary)
        gc.collect()
    return summaries

def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=root.parent / "generated_linear_test_mat",
        help="Directory containing healthData.mat / insoleData.mat / EMRData.mat.",
    )
    parser.add_argument(
        "--user-id",
        type=int,
        default=None,
        help="Single user id to run. If omitted, --user-ids is used.",
    )
    parser.add_argument(
        "--user-ids",
        default="1",
        help="Comma-separated user ids, e.g. 1 or 1,2,...,10.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=root / "old_matlab_equiv_runs",
        help="Base directory where per-user outputs are written.",
    )
    parser.add_argument(
        "--noise-preset",
        choices=sorted(NOISE_PRESETS.keys()),
        default="none",
        help="Column-aware noise preset applied to H1/H2 before running.",
    )
    parser.add_argument(
        "--noise-seed",
        type=int,
        default=20260423,
        help="Base seed used for deterministic per-user noise generation.",
    )
    parser.add_argument(
        "--noise-component-mode",
        choices=NOISE_COMPONENT_MODES,
        default="mixed",
        help=(
            "Which noise mechanism family to apply. 'mixed' keeps the current combined "
            "column-aware noise; 'laplace' uses heavy-tailed zero-mean noise; "
            "'impulse' uses sparse spike artifacts."
        ),
    )
    parser.add_argument(
        "--noise-strength-multiplier",
        type=float,
        default=1.0,
        help="Multiplier applied on top of the selected noise preset scales.",
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
        help="Reference estimator mode: use all 90 reference days, or sample repeated subsets.",
    )
    parser.add_argument(
        "--reference-sample-size",
        type=int,
        default=50,
        help="Number of reference days in each sampled SAA trial. Used only when --reference-sampling-mode sampled.",
    )
    parser.add_argument(
        "--reference-num-trials",
        type=int,
        default=20,
        help="Number of sampled SAA trials. Used only when --reference-sampling-mode sampled.",
    )
    parser.add_argument(
        "--reference-sampling-seed",
        type=int,
        default=20260423,
        help="Seed for deterministic reference-day sampling.",
    )
    parser.add_argument(
        "--history-window",
        type=int,
        default=DEFAULT_HISTORY_WINDOW,
        help="Number of previous/current time points L used in the second-stage history average.",
    )
    parser.add_argument(
        "--params-hat-rho",
        type=float,
        default=DEFAULT_PARAMS_HAT_RHO,
        help="Override for params_hat.rho used in the second-stage linear systems.",
    )
    parser.add_argument(
        "--params-hat-lambda",
        type=float,
        default=DEFAULT_PARAMS_HAT_LAMBDA,
        help="Override for params_hat.lambda_ used in the history/modal penalty term.",
    )
    parser.add_argument(
        "--params-hat-alpha-mix",
        default="0.75,0.60,0.15",
        help="Comma-separated override for params_hat.alpha_mix, e.g. 0.75,0.60,0.15.",
    )
    parser.add_argument(
        "--storage-dtype",
        choices=["float32", "float64"],
        default="float32",
        help="In-memory dtype for loaded H1/H2/H3/label arrays. float32 reduces memory substantially.",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=300,
        help="Print one progress line every N time steps. Use 1 to print every time step; use 0 to disable.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device to use. Default is cuda; use cpu for local smoke tests.",
    )
    parser.add_argument(
        "--torch-dtype",
        choices=["float32", "float64"],
        default="float64",
        help="Torch compute dtype. Use float64 for exact matching with the CPU implementation.",
    )
    parser.add_argument(
        "--target-day-batch-size",
        type=int,
        default=len(DAYS),
        help="Number of target days advanced together on the GPU.",
    )
    parser.add_argument(
        "--history-chunk-size",
        type=int,
        default=1024,
        help="Number of history time points processed per GPU chunk.",
    )
    parser.add_argument(
        "--verify-against-cpu",
        action="store_true",
        help="Run the CPU v2-equivalent path for each day and assert classification parity.",
    )
    args = parser.parse_args()

    if args.user_id is not None:
        user_ids = [args.user_id]
    else:
        user_ids = parse_int_list(args.user_ids)
    if args.target_day_batch_size <= 0:
        raise ValueError("--target-day-batch-size must be positive.")
    if args.history_chunk_size <= 0:
        raise ValueError("--history-chunk-size must be positive.")
    params_hat_alpha_mix = parse_float_tuple(
        args.params_hat_alpha_mix,
        expected_len=3,
        option_name="--params-hat-alpha-mix",
    )
    if args.params_hat_rho <= 0.0:
        raise ValueError("--params-hat-rho must be positive.")
    if args.params_hat_lambda < 0.0:
        raise ValueError("--params-hat-lambda must be non-negative.")

    noise_config = NoiseConfig(
        preset_name=args.noise_preset,
        seed=args.noise_seed,
        component_mode=args.noise_component_mode,
        strength_multiplier=args.noise_strength_multiplier,
        apply_scope=args.noise_apply_to,
    )
    reference_sampling_config = ReferenceSamplingConfig(
        mode=args.reference_sampling_mode,
        sample_size=args.reference_sample_size,
        num_trials=args.reference_num_trials,
        seed=args.reference_sampling_seed,
        history_window=args.history_window,
    )
    reference_sampling_config = effective_reference_sampling_config(
        reference_sampling_config,
        reference_pool_size=len(OTHERS),
    )
    print(
        "Noise config: "
        f"preset={noise_config.preset_name}, seed={noise_config.seed}, "
        f"component_mode={noise_config.component_mode}, "
        f"strength_multiplier={noise_config.strength_multiplier}, "
        f"apply_scope={noise_config.apply_scope}",
        flush=True,
    )
    print(
        "Reference sampling config: "
        f"mode={reference_sampling_config.mode}, "
        f"sample_size={reference_sampling_config.sample_size}, "
        f"num_trials={reference_sampling_config.num_trials}, "
        f"seed={reference_sampling_config.seed}, "
        f"history_window={reference_sampling_config.history_window}",
        flush=True,
    )
    storage_dtype = resolve_storage_dtype(args.storage_dtype)
    torch_dtype = resolve_torch_dtype(args.torch_dtype)
    device = resolve_torch_device(args.device)
    configure_torch_runtime()
    print(f"Storage dtype: {args.storage_dtype}", flush=True)
    print(f"Torch device: {device}", flush=True)
    print(f"Torch dtype: {args.torch_dtype}", flush=True)
    print(f"Target day batch size: {args.target_day_batch_size}", flush=True)
    print(f"History chunk size: {args.history_chunk_size}", flush=True)
    print(f"Verify against CPU: {args.verify_against_cpu}", flush=True)
    print(f"Progress interval: {args.progress_interval}", flush=True)
    print(
        "Model params: "
        f"rho={args.params_hat_rho}, "
        f"lambda_={args.params_hat_lambda}, "
        f"alpha_mix={list(params_hat_alpha_mix)}",
        flush=True,
    )

    summaries = run_users(
        data_dir=args.data_dir,
        user_ids=user_ids,
        out_root=args.out_root,
        noise_config=noise_config,
        reference_sampling_config=reference_sampling_config,
        storage_dtype=storage_dtype,
        progress_interval=args.progress_interval,
        device=device,
        torch_dtype=torch_dtype,
        target_day_batch_size=args.target_day_batch_size,
        history_chunk_size=args.history_chunk_size,
        verify_against_cpu=args.verify_against_cpu,
        params_hat_rho=args.params_hat_rho,
        params_hat_lambda=args.params_hat_lambda,
        params_hat_alpha_mix=params_hat_alpha_mix,
    )
    aggregate = aggregate_user_summaries(
        data_dir=args.data_dir,
        out_root=args.out_root,
        user_ids=user_ids,
        summaries=summaries,
    )
    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "all_users_summary.json").write_text(
        json.dumps(aggregate, indent=2),
        encoding="utf-8",
    )
    write_confusion_matrix_csv(
        args.out_root / "pooled_confusion_matrix.csv",
        np.asarray(aggregate["pooled_confusion_matrix"], dtype=np.int64),
    )
    write_aggregate_markdown_report(args.out_root / "aggregate_report.md", aggregate)
    if len(summaries) == 1:
        print(json.dumps(summaries[0], indent=2))
        return
    print(json.dumps(aggregate, indent=2))

if __name__ == "__main__":
    main()
