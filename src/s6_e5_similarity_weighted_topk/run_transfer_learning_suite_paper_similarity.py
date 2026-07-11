from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from _transfer_learning_common import (
    DEFAULTS,
    aggregate_transfer_summaries,
    build_similarity_csv_rows,
    cache_user_dir,
    ensure_dir,
    flatten_metric_record,
    load_response_cache,
    load_runner_module,
    parse_int_list,
    parse_k_values,
    parse_storage_dtype,
    run_transfer_inference_for_user,
    write_csv,
    write_json,
)

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
    storage_dtype = parse_storage_dtype(runner, storage_dtype_name)
    h_all, _labels_all = runner.load_user_tensors(data_dir, user_id, storage_dtype)
    return {
        "watch": np.asarray(h_all[0][reference_idx], dtype=storage_dtype),
        "insole": np.asarray(h_all[1][reference_idx], dtype=storage_dtype),
        "emr": np.asarray(h_all[2][reference_idx], dtype=storage_dtype),
    }

def pair_standardized_static_distance(
    target: np.ndarray,
    source: np.ndarray,
    *,
    eps: float,
) -> float:
    target64 = np.asarray(target, dtype=np.float64)
    source64 = np.asarray(source, dtype=np.float64)
    n_ref = int(target64.shape[0])
    if source64.shape != target64.shape:
        raise ValueError(f"Static modality shape mismatch: {target64.shape} vs {source64.shape}")
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
        raise ValueError("No time steps available for paper-similarity distance.")
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

def compute_paper_source_target_similarity(
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
        print(f"[paper-similarity] loading user {user_id:02d}", flush=True)
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
        print(f"[paper-similarity] target user {target_user_id:02d}", flush=True)
        target_features = features_by_user[target_user_id]
        target_rows: list[dict[str, Any]] = []
        for source_user_id in source_user_ids:
            if source_user_id == target_user_id:
                continue
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
                "target_user_id": target_user_id,
                "source_user_id": source_user_id,
                "distance": distance,
                "d_emr": float(d_emr),
                "d_watch": float(d_watch),
                "d_insole": float(d_insole),
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
    meta = {
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

def main() -> None:
    runner = load_runner_module()
    parser = argparse.ArgumentParser(
        description="Run transfer learning with the matrix Frobenius source-target similarity described in the manuscript."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULTS.data_dir)
    parser.add_argument("--baseline-root", type=Path, default=DEFAULTS.baseline_root)
    parser.add_argument("--response-cache-root", type=Path, default=DEFAULTS.cache_root)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--target-user-ids", default="11,12,13,14,15,16,17,18,19,20")
    parser.add_argument("--candidate-user-ids", default="1,2,3,4,5,6,7,8,9,10")
    parser.add_argument("--method2-k-values", default="1,2,3,4,5,6,7,8,9,10")
    parser.add_argument("--storage-dtype", default=DEFAULTS.storage_dtype, choices=["float32", "float64"])
    parser.add_argument(
        "--reference-mode",
        default="first90",
        choices=["first90", "runner_others"],
        help="Historical rows used in the similarity calculation. 'first90' follows the manuscript formula.",
    )
    parser.add_argument("--similarity-chunk-size", type=int, default=256)
    parser.add_argument("--similarity-time-step-limit", type=int, default=None)
    parser.add_argument("--time-step-limit", type=int, default=None)
    parser.add_argument("--eps", type=float, default=1e-12)
    args = parser.parse_args()

    target_user_ids = parse_int_list(args.target_user_ids)
    candidate_user_ids = parse_int_list(args.candidate_user_ids)
    k_values = parse_k_values(args.method2_k_values)
    if args.similarity_chunk_size <= 0:
        raise ValueError("--similarity-chunk-size must be positive.")
    if args.similarity_time_step_limit is not None and args.similarity_time_step_limit <= 0:
        raise ValueError("--similarity-time-step-limit must be positive when provided.")
    if args.time_step_limit is not None and args.time_step_limit <= 0:
        raise ValueError("--time-step-limit must be positive when provided.")

    ensure_dir(args.output_root)
    pair_rows, ranking_by_target, similarity_meta = compute_paper_source_target_similarity(
        runner,
        data_dir=args.data_dir,
        target_user_ids=target_user_ids,
        source_user_ids=candidate_user_ids,
        storage_dtype_name=args.storage_dtype,
        reference_mode=args.reference_mode,
        chunk_size=args.similarity_chunk_size,
        time_step_limit=args.similarity_time_step_limit,
        eps=args.eps,
    )
    write_json(args.output_root / "pairwise_similarity.json", pair_rows)
    write_csv(args.output_root / "pairwise_similarity.csv", build_similarity_csv_rows(pair_rows))
    write_json(args.output_root / "similarity_config.json", similarity_meta)

    method1_pair_summaries: list[dict[str, Any]] = []
    method1_pair_records: list[dict[str, Any]] = []
    method1_nearest_summaries: list[dict[str, Any]] = []
    method2_records: list[dict[str, Any]] = []
    method2_summaries_by_k: dict[int, list[dict[str, Any]]] = {k: [] for k in k_values}

    for target_user_id in target_user_ids:
        ranked_sources = list(ranking_by_target[target_user_id])
        if max(k_values) > len(ranked_sources):
            raise ValueError(f"Target {target_user_id} has fewer sources than requested K values.")
        distances = [float(item["distance"]) for item in ranked_sources]
        tau = max(float(np.median(np.asarray(distances, dtype=np.float64))), 1e-12)
        exp_distances = [math.exp(-distance / tau) for distance in distances]
        cum_y1 = None
        cum_y2 = None
        cum_y3 = None
        cumulative_weight = 0.0
        nearest_summary = None

        print(
            f"=== Target user {target_user_id:02d}: paper-similarity sources={len(ranked_sources)}, "
            f"tau={tau:.6g} ===",
            flush=True,
        )
        for rank_idx, source_row in enumerate(ranked_sources, start=1):
            source_user_id = int(source_row["source_user_id"])
            source_distance = float(source_row["distance"])
            raw_weight = float(exp_distances[rank_idx - 1])
            cache = load_response_cache(args.response_cache_root, source_user_id)
            if not np.array_equal(cache["days"], runner.DAYS.astype(np.int64)):
                raise ValueError(
                    f"Response cache days mismatch for source user {source_user_id}: "
                    f"{cache_user_dir(args.response_cache_root, source_user_id) / 'response_cache.npz'}"
                )

            y1 = np.asarray(cache["y1_mean"], dtype=np.float64)
            y2 = np.asarray(cache["y2_mean"], dtype=np.float64)
            y3 = np.asarray(cache["y3_mean"], dtype=np.float64)

            method1_summary = run_transfer_inference_for_user(
                runner,
                data_dir=args.data_dir,
                user_id=target_user_id,
                transferred_y1=y1,
                transferred_y2=y2,
                transferred_y3=y3,
                method="method1_frozen_response_transfer_paper_similarity",
                source_user_ids=[source_user_id],
                source_distances=[source_distance],
                weights=[1.0],
                tau=None,
                k=1,
                storage_dtype_name=args.storage_dtype,
                time_step_limit=args.time_step_limit,
            )
            method1_pair_summaries.append(method1_summary)
            method1_pair_records.append(dict(method1_summary))
            if rank_idx == 1:
                nearest_summary = method1_summary

            if cum_y1 is None:
                cum_y1 = raw_weight * y1
                cum_y2 = raw_weight * y2
                cum_y3 = raw_weight * y3
            else:
                cum_y1 += raw_weight * y1
                cum_y2 += raw_weight * y2
                cum_y3 += raw_weight * y3
            cumulative_weight += raw_weight

            if rank_idx in k_values:
                weighted_sources = ranked_sources[:rank_idx]
                normalized_weights = [
                    float(exp_distances[idx] / cumulative_weight) for idx in range(rank_idx)
                ]
                assert cum_y1 is not None and cum_y2 is not None and cum_y3 is not None
                method2_summary = run_transfer_inference_for_user(
                    runner,
                    data_dir=args.data_dir,
                    user_id=target_user_id,
                    transferred_y1=cum_y1 / cumulative_weight,
                    transferred_y2=cum_y2 / cumulative_weight,
                    transferred_y3=cum_y3 / cumulative_weight,
                    method="method2_similarity_weighted_response_transfer_paper_similarity",
                    source_user_ids=[int(item["source_user_id"]) for item in weighted_sources],
                    source_distances=[float(item["distance"]) for item in weighted_sources],
                    weights=normalized_weights,
                    tau=tau,
                    k=rank_idx,
                    storage_dtype_name=args.storage_dtype,
                    time_step_limit=args.time_step_limit,
                )
                method2_summaries_by_k[rank_idx].append(method2_summary)
                method2_records.append(dict(method2_summary))

        if nearest_summary is None:
            raise RuntimeError(f"Unable to determine nearest summary for target {target_user_id}.")
        method1_nearest_summaries.append(nearest_summary)

    write_json(args.output_root / "method1_all_pairs.json", method1_pair_records)
    write_csv(
        args.output_root / "method1_all_pairs.csv",
        [flatten_metric_record(summary) for summary in method1_pair_summaries],
    )

    method1_nearest_aggregate = aggregate_transfer_summaries(
        runner,
        data_dir=args.data_dir,
        out_root=args.output_root / "method1_nearest",
        user_ids=target_user_ids,
        summaries=method1_nearest_summaries,
    )
    write_json(args.output_root / "method1_nearest_summary.json", method1_nearest_aggregate)
    write_csv(
        args.output_root / "method1_nearest_summary.csv",
        [flatten_metric_record(summary) for summary in method1_nearest_summaries],
    )

    write_json(args.output_root / "method2_k_sweep.json", method2_records)
    write_csv(
        args.output_root / "method2_k_sweep.csv",
        [flatten_metric_record(summary) for summary in method2_records],
    )

    best_candidates: list[dict[str, Any]] = []
    method2_k_pooled_rows: list[dict[str, Any]] = []
    for k_value in sorted(k_values):
        summaries = method2_summaries_by_k[k_value]
        aggregate = aggregate_transfer_summaries(
            runner,
            data_dir=args.data_dir,
            out_root=args.output_root / f"method2_k_{k_value}",
            user_ids=target_user_ids,
            summaries=summaries,
        )
        pooled = aggregate["pooled_metrics"]
        row = {
            "k": k_value,
            "macro_f1_pooled": float(pooled["macro_f1"]),
            "balanced_accuracy_pooled": float(pooled["balanced_accuracy"]),
            "accuracy_percent_pooled": float(pooled["accuracy_percent"]),
            "weighted_f1_pooled": float(pooled["weighted_f1"]),
            "worst_class_recall_pooled": float(pooled["worst_class_recall"]),
            "worst_class_f1_pooled": float(pooled["worst_class_f1"]),
            "macro_f1_mean": float(np.mean([float(summary["macro_f1"]) for summary in summaries])),
            "balanced_accuracy_mean": float(
                np.mean([float(summary["balanced_accuracy"]) for summary in summaries])
            ),
            "accuracy_percent_mean": float(
                np.mean([float(summary["accuracy_percent"]) for summary in summaries])
            ),
        }
        method2_k_pooled_rows.append(row)
        best_candidates.append(
            dict(row)
        )
    write_json(args.output_root / "method2_k_pooled_metrics.json", method2_k_pooled_rows)
    write_csv(args.output_root / "method2_k_pooled_metrics.csv", method2_k_pooled_rows)
    best_candidates.sort(
        key=lambda item: (
            -float(item["macro_f1_pooled"]),
            -float(item["balanced_accuracy_pooled"]),
            -float(item["accuracy_percent_pooled"]),
            int(item["k"]),
        )
    )
    selected_k = int(best_candidates[0]["k"])
    best_k_summaries = method2_summaries_by_k[selected_k]
    method2_best_aggregate = aggregate_transfer_summaries(
        runner,
        data_dir=args.data_dir,
        out_root=args.output_root / "method2_best_k",
        user_ids=target_user_ids,
        summaries=best_k_summaries,
    )
    method2_best_aggregate["selected_k"] = selected_k
    write_json(args.output_root / "method2_best_k_summary.json", method2_best_aggregate)
    write_csv(
        args.output_root / "method2_best_k_summary.csv",
        [flatten_metric_record(summary) for summary in best_k_summaries],
    )
    write_json(
        args.output_root / "best_k_report.json",
        {
            "baseline_root": str(args.baseline_root),
            "response_cache_root": str(args.response_cache_root),
            "output_root": str(args.output_root),
            "target_user_ids": target_user_ids,
            "candidate_user_ids": candidate_user_ids,
            "time_step_limit": args.time_step_limit,
            "similarity": similarity_meta,
            "selection_metric_source": "pooled_confusion_matrix",
            "selected_k": selected_k,
            "ranked_candidates": best_candidates,
            "selected_k_user_details": [
                {
                    "user_id": int(summary["user_id"]),
                    "source_user_ids": summary["transfer"]["source_user_ids"],
                    "source_distances": summary["transfer"]["source_distances"],
                    "weights": summary["transfer"]["weights"],
                    "tau": summary["transfer"]["tau"],
                    "accuracy_percent": summary["accuracy_percent"],
                    "macro_f1": summary["macro_f1"],
                    "balanced_accuracy": summary["balanced_accuracy"],
                }
                for summary in best_k_summaries
            ],
        },
    )

if __name__ == "__main__":
    main()
