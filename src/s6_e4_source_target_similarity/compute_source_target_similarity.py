#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

RUNNER_MODULE_NAME = "_s6_e4_runner_old_matlab_equivalent"
RUNNER_PATH = Path(__file__).resolve().parent / "run_old_matlab_equivalent.py"

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
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one user id.")
    return values

def reference_indices(runner, mode: str) -> np.ndarray:
    if mode == "first90":
        return np.arange(90, dtype=np.int64)
    if mode == "runner_others":
        return runner.OTHERS.astype(np.int64) - 1
    raise ValueError(f"Unsupported reference mode: {mode}")

def load_reference_modalities(
    runner,
    *,
    data_dir: Path,
    user_id: int,
    reference_idx: np.ndarray,
    storage_dtype_name: str,
) -> dict[str, np.ndarray]:
    storage_dtype = runner.resolve_storage_dtype(storage_dtype_name)
    h_all, _labels_all = runner.load_user_tensors(data_dir, user_id, storage_dtype)
    return {
        "watch": np.asarray(h_all[0][reference_idx], dtype=storage_dtype),
        "insole": np.asarray(h_all[1][reference_idx], dtype=storage_dtype),
        "emr": np.asarray(h_all[2][reference_idx], dtype=storage_dtype),
    }

def pair_standardized_static_distance(target: np.ndarray, source: np.ndarray, *, eps: float) -> float:
    target64 = np.asarray(target, dtype=np.float64)
    source64 = np.asarray(source, dtype=np.float64)
    if source64.shape != target64.shape:
        raise ValueError(f"Static modality shape mismatch: {target64.shape} vs {source64.shape}")
    n_ref = int(target64.shape[0])
    mu = (np.sum(target64, axis=0) + np.sum(source64, axis=0)) / float(2 * n_ref)
    var = (
        np.sum(np.square(target64 - mu[None, :]), axis=0)
        + np.sum(np.square(source64 - mu[None, :]), axis=0)
    ) / float(2 * n_ref)
    sigma = np.sqrt(np.maximum(var, eps))
    diff = (target64 - mu[None, :]) / sigma[None, :] - (source64 - mu[None, :]) / sigma[None, :]
    return float(np.sum(np.square(diff)) / float(n_ref))

def pair_standardized_dynamic_distance(
    target: np.ndarray,
    source: np.ndarray,
    *,
    chunk_size: int,
    time_step_limit: int | None,
    eps: float,
) -> float:
    if target.shape[:2] != source.shape[:2]:
        raise ValueError(f"Dynamic modality shape mismatch: {target.shape} vs {source.shape}")
    n_ref = int(target.shape[0])
    t_day = min(int(target.shape[2]), int(source.shape[2]))
    if time_step_limit is not None:
        t_day = min(t_day, int(time_step_limit))
    if t_day <= 0:
        raise ValueError("No time steps available for similarity distance.")

    total = 0.0
    for start in range(0, t_day, chunk_size):
        stop = min(start + chunk_size, t_day)
        target_chunk = np.asarray(target[:, :, start:stop], dtype=np.float64)
        source_chunk = np.asarray(source[:, :, start:stop], dtype=np.float64)
        mu = (
            np.sum(target_chunk, axis=0, keepdims=True)
            + np.sum(source_chunk, axis=0, keepdims=True)
        ) / float(2 * n_ref)
        var = (
            np.sum(np.square(target_chunk - mu), axis=0, keepdims=True)
            + np.sum(np.square(source_chunk - mu), axis=0, keepdims=True)
        ) / float(2 * n_ref)
        sigma = np.sqrt(np.maximum(var, eps))
        diff = (target_chunk - mu) / sigma - (source_chunk - mu) / sigma
        total += float(np.sum(np.square(diff)))
    return total / float(t_day * n_ref)

def compute_similarity(
    runner,
    *,
    data_dir: Path,
    target_user_ids: list[int],
    source_user_ids: list[int],
    storage_dtype_name: str,
    reference_mode: str,
    chunk_size: int,
    time_step_limit: int | None,
    eps: float,
) -> tuple[list[dict[str, Any]], dict[int, list[dict[str, Any]]], dict[str, Any]]:
    reference_idx = reference_indices(runner, reference_mode)
    all_user_ids = sorted(set(target_user_ids) | set(source_user_ids))
    features_by_user: dict[int, dict[str, np.ndarray]] = {}

    load_started = time.time()
    for user_id in all_user_ids:
        print(f"[S6-E4] loading user {user_id:02d}", flush=True)
        features_by_user[user_id] = load_reference_modalities(
            runner,
            data_dir=data_dir,
            user_id=user_id,
            reference_idx=reference_idx,
            storage_dtype_name=storage_dtype_name,
        )
    load_seconds = time.time() - load_started

    pair_rows: list[dict[str, Any]] = []
    ranking_by_target: dict[int, list[dict[str, Any]]] = {}
    distance_started = time.time()
    for target_user_id in target_user_ids:
        print(f"[S6-E4] computing target user {target_user_id:02d}", flush=True)
        target_features = features_by_user[target_user_id]
        target_rows: list[dict[str, Any]] = []
        for source_user_id in source_user_ids:
            source_features = features_by_user[source_user_id]
            d_watch = pair_standardized_dynamic_distance(
                target_features["watch"],
                source_features["watch"],
                chunk_size=chunk_size,
                time_step_limit=time_step_limit,
                eps=eps,
            )
            d_insole = pair_standardized_dynamic_distance(
                target_features["insole"],
                source_features["insole"],
                chunk_size=chunk_size,
                time_step_limit=time_step_limit,
                eps=eps,
            )
            d_emr = pair_standardized_static_distance(
                target_features["emr"],
                source_features["emr"],
                eps=eps,
            )
            distance = float(2.0 * d_watch + 2.0 * d_insole + d_emr)
            row = {
                "target_user_id": int(target_user_id),
                "source_user_id": int(source_user_id),
                "distance": distance,
                "d_emr": float(d_emr),
                "d_watch": float(d_watch),
                "d_insole": float(d_insole),
            }
            pair_rows.append(row)
            target_rows.append(row)

        target_rows.sort(key=lambda item: (float(item["distance"]), int(item["source_user_id"])))
        distances = [float(row["distance"]) for row in target_rows]
        tau = max(float(np.median(np.asarray(distances, dtype=np.float64))), eps)
        kernels = [math.exp(-distance / tau) for distance in distances]
        kernel_sum = max(float(sum(kernels)), eps)
        for rank, (row, kernel) in enumerate(zip(target_rows, kernels), start=1):
            row["rank"] = int(rank)
            row["tau"] = tau
            row["kernel"] = float(kernel)
            row["normalized_weight"] = float(kernel / kernel_sum)
        ranking_by_target[target_user_id] = target_rows

    pair_rows.sort(
        key=lambda item: (
            int(item["target_user_id"]),
            int(item["rank"]),
            int(item["source_user_id"]),
        )
    )
    meta = {
        "experiment": "S6-E4",
        "similarity": "paper_matrix_frobenius",
        "reference_mode": reference_mode,
        "reference_indices_1based": [int(value) + 1 for value in reference_idx.tolist()],
        "chunk_size": chunk_size,
        "time_step_limit": time_step_limit,
        "standardization": "pairwise pooled column standardization for each time slice",
        "gamma": {"watch": 2.0, "insole": 2.0, "emr": 1.0},
        "load_seconds": load_seconds,
        "distance_seconds": time.time() - distance_started,
    }
    return pair_rows, ranking_by_target, meta

def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})

def build_target_summary(ranking_by_target: dict[int, list[dict[str, Any]]], top_k: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target_user_id, ranked_rows in sorted(ranking_by_target.items()):
        top_rows = ranked_rows[:top_k]
        rows.append(
            {
                "target_user_id": int(target_user_id),
                "tau": float(ranked_rows[0]["tau"]),
                "nearest_source_user_id": int(ranked_rows[0]["source_user_id"]),
                "nearest_distance": float(ranked_rows[0]["distance"]),
                "top_k": int(top_k),
                "top_k_source_user_ids": ",".join(f"{int(row['source_user_id']):02d}" for row in top_rows),
                "top_k_distances": ";".join(f"{float(row['distance']):.10g}" for row in top_rows),
                "top_k_all_source_normalized_weights": ";".join(
                    f"{float(row['normalized_weight']):.10g}" for row in top_rows
                ),
            }
        )
    return rows

def write_markdown(
    path: Path,
    *,
    meta: dict[str, Any],
    target_summary_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# S6-E4 Source-Target Similarity",
        "",
        f"- similarity: `{meta['similarity']}`",
        f"- reference mode: `{meta['reference_mode']}`",
        f"- gamma: `watch=2, insole=2, emr=1`",
        f"- distance seconds: `{float(meta['distance_seconds']):.3f}`",
        "",
        "| target user | tau | nearest source | nearest distance | top-K sources |",
        "| ---: | ---: | ---: | ---: | --- |",
    ]
    for row in target_summary_rows:
        lines.append(
            f"| {int(row['target_user_id'])} | "
            f"{float(row['tau']):.6f} | "
            f"{int(row['nearest_source_user_id'])} | "
            f"{float(row['nearest_distance']):.6f} | "
            f"{row['top_k_source_user_ids']} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute S6-E4 source-target user similarity and transfer weights."
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--target-user-ids", default="11,12,13,14,15,16,17,18,19,20")
    parser.add_argument("--source-user-ids", default="1,2,3,4,5,6,7,8,9,10")
    parser.add_argument("--storage-dtype", default="float32", choices=["float32", "float64"])
    parser.add_argument("--reference-mode", default="first90", choices=["first90", "runner_others"])
    parser.add_argument("--similarity-chunk-size", type=int, default=256)
    parser.add_argument("--similarity-time-step-limit", type=int, default=None)
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument("--summary-top-k", type=int, default=8)
    args = parser.parse_args()

    if args.similarity_chunk_size <= 0:
        raise ValueError("--similarity-chunk-size must be positive.")
    if args.similarity_time_step_limit is not None and args.similarity_time_step_limit <= 0:
        raise ValueError("--similarity-time-step-limit must be positive when provided.")
    if args.summary_top_k <= 0:
        raise ValueError("--summary-top-k must be positive.")

    runner = load_runner_module()
    target_user_ids = parse_int_list(args.target_user_ids)
    source_user_ids = parse_int_list(args.source_user_ids)
    if args.summary_top_k > len(source_user_ids):
        raise ValueError("--summary-top-k cannot exceed the number of source users.")

    args.output_root.mkdir(parents=True, exist_ok=True)
    pair_rows, ranking_by_target, meta = compute_similarity(
        runner,
        data_dir=args.data_dir,
        target_user_ids=target_user_ids,
        source_user_ids=source_user_ids,
        storage_dtype_name=args.storage_dtype,
        reference_mode=args.reference_mode,
        chunk_size=args.similarity_chunk_size,
        time_step_limit=args.similarity_time_step_limit,
        eps=args.eps,
    )

    target_summary_rows = build_target_summary(ranking_by_target, args.summary_top_k)
    pairwise_fields = [
        "target_user_id",
        "source_user_id",
        "rank",
        "distance",
        "d_emr",
        "d_watch",
        "d_insole",
    ]
    weight_fields = pairwise_fields + ["tau", "kernel", "normalized_weight"]
    write_json(args.output_root / "pairwise_similarity.json", pair_rows)
    write_csv(args.output_root / "pairwise_similarity.csv", pair_rows, pairwise_fields)
    write_csv(args.output_root / "source_target_transfer_weights.csv", pair_rows, weight_fields)
    write_csv(
        args.output_root / "target_similarity_summary.csv",
        target_summary_rows,
        [
            "target_user_id",
            "tau",
            "nearest_source_user_id",
            "nearest_distance",
            "top_k",
            "top_k_source_user_ids",
            "top_k_distances",
            "top_k_all_source_normalized_weights",
        ],
    )
    write_json(args.output_root / "similarity_config.json", meta)
    write_markdown(args.output_root / "summary.md", meta=meta, target_summary_rows=target_summary_rows)
    print(json.dumps({"meta": meta, "target_summary": target_summary_rows}, indent=2))

if __name__ == "__main__":
    main()
