#!/usr/bin/env python3
"""
Generate the legacy Linear Test .mat files from DSVI-O-main/dataset.

This script is intentionally strict by default:
- it uses the historical categorical encodings found in
  DSVI-O-main/Linear Test_all_person
- it reproduces the legacy shared EMR layout
- it verifies the generated .mat payloads against the reference directory
  by default

For new source/target transfer experiments, use --skip-verify and
--emr-layout per_user_repeated so newly generated users can be converted
without depending on the legacy 1..10 reference layout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, NamedTuple

import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat

HEALTH_EXCLUDE_COLUMNS = {"time", "status", "abnormal_group"}
INSOLE_EXCLUDE_COLUMNS = {"time"}
EMR_EXCLUDE_COLUMNS = {"PatientID"}

class DatasetLayout(NamedTuple):
    name: str
    health_dir: Path
    insole_dir: Path
    emr_dir: Path

DATASET_LAYOUTS = (
    DatasetLayout(
        name="legacy",
        health_dir=Path("physiological_data"),
        insole_dir=Path("insole_data"),
        emr_dir=Path("clinical_indicators"),
    ),
    DatasetLayout(
        name="generated",
        health_dir=Path("generated_data") / "health_data",
        insole_dir=Path("generated_data") / "insole_data",
        emr_dir=Path("generated_data") / "electronic_medical_records",
    ),
)

HEALTH_CODEBOOK = {
    "activity::Active": 1.0,
    "activity::Resting": 2.0,
    "activity::Waking": 3.0,
    "activity::Sleeping": 4.0,
    "activity::Dinner": 5.0,
    "activity::Lunch": 6.0,
    "activity::Leisure": 7.0,
}

INSOLE_CODEBOOK = {
    "gait_type::Brisk": 0.0,
    "gait_type::Abnormal": 1.0,
    "gait_type::Normal": 2.0,
    "gait_type::Resting": 3.0,
    "gait_type::Slow": 4.0,
    "motion_type::Stationary": 1.0,
    "motion_type::Fast Walking": 2.0,
    "motion_type::Normal Walking": 3.0,
    "motion_type::Slow Walking": 4.0,
    "motion_type::Running": 5.0,
}

CLINICAL_CODEBOOK = {
    "Gender::female": 1.0,
    "Gender::male": 2.0,
}

GROUP_CODEBOOKS = {
    "health": HEALTH_CODEBOOK,
    "insole": INSOLE_CODEBOOK,
}

def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    dsvi_root = script_dir.parent

    parser = argparse.ArgumentParser(
        description="Generate legacy Linear Test .mat files from DSVI-O-main/dataset."
    )
    parser.add_argument(
        "--dataset-root",
        default=str(dsvi_root / "dataset"),
        help="Directory containing v1..v10 dataset folders.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(script_dir / "generated_linear_test_mat"),
        help="Where to write healthData.mat / insoleData.mat / EMRData.mat.",
    )
    parser.add_argument(
        "--reference-dir",
        default=str(dsvi_root / "Linear Test_all_person"),
        help="Reference .mat directory used for equality verification.",
    )
    parser.add_argument(
        "--versions",
        default="v1,v2,v3,v4,v5,v6,v7,v8,v9,v10",
        help="Comma-separated version order for concatenation.",
    )
    parser.add_argument(
        "--user-ids",
        default="1,2,3,4,5,6,7,8,9,10",
        help="Comma-separated user ids to convert.",
    )
    parser.add_argument(
        "--samples-per-day",
        type=int,
        default=17280,
        help="Expected 5-second samples per day.",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip comparison against the reference .mat files after writing.",
    )
    parser.add_argument(
        "--emr-layout",
        choices=["legacy_shared_cycle", "per_user_repeated"],
        default="legacy_shared_cycle",
        help=(
            "legacy_shared_cycle reproduces the original all-person MAT layout. "
            "per_user_repeated writes each user's own EMR row repeated for that user's 100 generated days."
        ),
    )
    return parser.parse_args()

def parse_csv_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]

def parse_int_list(raw: str) -> list[int]:
    return [int(item) for item in parse_csv_list(raw)]

def resolve_versions(dataset_root: Path, requested: Iterable[str]) -> list[str]:
    versions = list(requested)
    missing = [version for version in versions if not (dataset_root / version).is_dir()]
    if missing:
        raise FileNotFoundError(f"Missing version directories under {dataset_root}: {missing}")
    return versions

def detect_dataset_layout(version_dir: Path, probe_user_id: int) -> DatasetLayout:
    for layout in DATASET_LAYOUTS:
        if (
            (version_dir / layout.health_dir / f"{probe_user_id}_health_data.csv").exists()
            and (version_dir / layout.insole_dir / f"{probe_user_id}_insole_data.csv").exists()
            and (version_dir / layout.emr_dir / f"{probe_user_id}_EMR.csv").exists()
        ):
            return layout

    expected = []
    for layout in DATASET_LAYOUTS:
        expected.extend(
            [
                version_dir / layout.health_dir / f"{probe_user_id}_health_data.csv",
                version_dir / layout.insole_dir / f"{probe_user_id}_insole_data.csv",
                version_dir / layout.emr_dir / f"{probe_user_id}_EMR.csv",
            ]
        )
    raise FileNotFoundError(
        "Could not detect a supported dataset layout under "
        f"{version_dir}. Expected one of: {[str(path) for path in expected]}"
    )

def health_path(version_dir: Path, layout: DatasetLayout, user_id: int) -> Path:
    return version_dir / layout.health_dir / f"{user_id}_health_data.csv"

def insole_path(version_dir: Path, layout: DatasetLayout, user_id: int) -> Path:
    return version_dir / layout.insole_dir / f"{user_id}_insole_data.csv"

def emr_path(version_dir: Path, layout: DatasetLayout, user_id: int) -> Path:
    return version_dir / layout.emr_dir / f"{user_id}_EMR.csv"

def read_feature_columns(
    version_dir: Path,
    layout: DatasetLayout,
    probe_user_id: int,
) -> tuple[list[str], list[str], list[str]]:
    health_file = health_path(version_dir, layout, probe_user_id)
    insole_file = insole_path(version_dir, layout, probe_user_id)
    emr_file = emr_path(version_dir, layout, probe_user_id)

    health_columns = [
        column for column in pd.read_csv(health_file, nrows=0).columns if column not in HEALTH_EXCLUDE_COLUMNS
    ]
    insole_columns = [
        column for column in pd.read_csv(insole_file, nrows=0).columns if column not in INSOLE_EXCLUDE_COLUMNS
    ]
    emr_columns = [
        column for column in pd.read_csv(emr_file, nrows=0).columns if column not in EMR_EXCLUDE_COLUMNS
    ]
    return health_columns, insole_columns, emr_columns

def encode_categorical(group: str, column: str, value: object) -> float:
    key = f"{column}::{value}"
    try:
        return GROUP_CODEBOOKS[group][key]
    except KeyError as exc:
        raise ValueError(f"Unsupported categorical value for {group}.{column}: {value!r}") from exc

def frame_to_numeric_matrix(
    df: pd.DataFrame,
    columns: list[str],
    *,
    categorical_columns: set[str],
    group: str,
) -> np.ndarray:
    numeric_columns: list[np.ndarray] = []
    for column in columns:
        if column in categorical_columns:
            encoded = [encode_categorical(group, column, value) for value in df[column].astype(str)]
            numeric_columns.append(np.asarray(encoded, dtype=np.float64))
        else:
            numeric_columns.append(pd.to_numeric(df[column], errors="raise").to_numpy(dtype=np.float64))
    return np.column_stack(numeric_columns).astype(np.float64, copy=False)

def read_emr_vector(emr_path: Path, emr_columns: list[str]) -> np.ndarray:
    row = pd.read_csv(emr_path).iloc[0]
    values: list[float] = []
    for column in emr_columns:
        if column == "Gender":
            gender = str(row[column]).strip().lower()
            try:
                values.append(CLINICAL_CODEBOOK[f"Gender::{gender}"])
            except KeyError as exc:
                raise ValueError(f"Unsupported Gender value in {emr_path}: {row[column]!r}") from exc
        else:
            values.append(float(row[column]))
    return np.asarray(values, dtype=np.float64)

def split_days(matrix: np.ndarray, samples_per_day: int, source_path: Path) -> np.ndarray:
    if matrix.shape[0] % samples_per_day != 0:
        raise ValueError(
            f"{source_path} has {matrix.shape[0]} rows, not divisible by samples_per_day={samples_per_day}"
        )
    return matrix.reshape(matrix.shape[0] // samples_per_day, samples_per_day, *matrix.shape[1:])

def validate_aligned_time(
    health_df: pd.DataFrame,
    insole_df: pd.DataFrame,
    health_path: Path,
    insole_path: Path,
) -> None:
    if len(health_df) != len(insole_df):
        raise ValueError(f"Row count mismatch: {health_path} vs {insole_path}")
    if not health_df["time"].equals(insole_df["time"]):
        mismatch = np.flatnonzero((health_df["time"] != insole_df["time"]).to_numpy())
        first = int(mismatch[0]) if len(mismatch) else -1
        raise ValueError(
            f"Timestamp mismatch between {health_path} and {insole_path} at row {first}"
        )

def build_shared_emr_matrix(
    *,
    dataset_root: Path,
    layout: DatasetLayout,
    versions: list[str],
    user_ids: list[int],
    emr_columns: list[str],
) -> np.ndarray:
    rows: list[np.ndarray] = []
    for version in versions:
        version_dir = dataset_root / version
        for user_id in user_ids:
            rows.append(read_emr_vector(emr_path(version_dir, layout, user_id), emr_columns))
    return np.stack(rows, axis=0).astype(np.float64, copy=False)

def build_per_user_repeated_emr_matrix(
    *,
    dataset_root: Path,
    layout: DatasetLayout,
    emr_columns: list[str],
    user_id: int,
    row_meta: list[dict[str, object]],
) -> np.ndarray:
    vectors_by_version: dict[str, np.ndarray] = {}
    rows: list[np.ndarray] = []
    for row in row_meta:
        version = str(row["version"])
        if version not in vectors_by_version:
            vectors_by_version[version] = read_emr_vector(
                emr_path(dataset_root / version, layout, user_id),
                emr_columns,
            )
        rows.append(vectors_by_version[version])
    return np.stack(rows, axis=0).astype(np.float64, copy=False)

def build_user_tensors(
    *,
    dataset_root: Path,
    layout: DatasetLayout,
    versions: list[str],
    user_id: int,
    health_columns: list[str],
    insole_columns: list[str],
    samples_per_day: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, object]]]:
    health_days: list[np.ndarray] = []
    insole_days: list[np.ndarray] = []
    label_days: list[np.ndarray] = []
    row_meta: list[dict[str, object]] = []

    for version in versions:
        version_dir = dataset_root / version
        health_file = health_path(version_dir, layout, user_id)
        insole_file = insole_path(version_dir, layout, user_id)
        emr_file = emr_path(version_dir, layout, user_id)
        if not health_file.exists() or not insole_file.exists() or not emr_file.exists():
            raise FileNotFoundError(f"Missing files for user {user_id} in {version_dir}")

        health_df = pd.read_csv(health_file)
        insole_df = pd.read_csv(insole_file)
        validate_aligned_time(health_df, insole_df, health_file, insole_file)

        health_matrix = frame_to_numeric_matrix(
            health_df,
            health_columns,
            categorical_columns={"activity"},
            group="health",
        )
        insole_matrix = frame_to_numeric_matrix(
            insole_df,
            insole_columns,
            categorical_columns={"gait_type", "motion_type"},
            group="insole",
        )
        labels = pd.to_numeric(health_df["status"], errors="raise").to_numpy(dtype=np.float64)

        health_split = split_days(health_matrix, samples_per_day, health_file)
        insole_split = split_days(insole_matrix, samples_per_day, insole_file)
        label_split = split_days(labels[:, None], samples_per_day, health_file)[:, :, 0]

        if not (health_split.shape[0] == insole_split.shape[0] == label_split.shape[0]):
            raise ValueError(f"Day count mismatch for user {user_id}, version {version}")

        for day_index in range(health_split.shape[0]):
            health_days.append(health_split[day_index])
            insole_days.append(insole_split[day_index])
            label_days.append(label_split[day_index])
            row_meta.append(
                {
                    "row_1based": len(row_meta) + 1,
                    "user_id": user_id,
                    "version": version,
                    "day_in_version_1based": day_index + 1,
                    "date": str(health_df["time"].iloc[day_index * samples_per_day])[:10],
                }
            )

    return (
        np.stack(health_days, axis=0).astype(np.float64, copy=False),
        np.stack(insole_days, axis=0).astype(np.float64, copy=False),
        np.stack(label_days, axis=0).astype(np.float64, copy=False),
        row_meta,
    )

def write_manifest(
    *,
    output_dir: Path,
    dataset_root: Path,
    layout: DatasetLayout,
    versions: list[str],
    user_ids: list[int],
    health_columns: list[str],
    insole_columns: list[str],
    emr_columns: list[str],
    emr_layout: str,
    samples_per_day: int,
    row_meta_by_user: dict[str, list[dict[str, object]]],
) -> None:
    manifest = {
        "source": str(dataset_root),
        "dataset_layout": {
            "name": layout.name,
            "health_dir": str(layout.health_dir),
            "insole_dir": str(layout.insole_dir),
            "clinical_indicators_dir": str(layout.emr_dir),
        },
        "versions": versions,
        "user_ids": user_ids,
        "row_order": "For each user: v1 day 1..10, v2 day 1..10, ..., v10 day 1..10.",
        "emr_layout": emr_layout,
        "dtype": "float64",
        "samples_per_day": samples_per_day,
        "feature_names": {
            "health": health_columns,
            "insole": insole_columns,
            "clinical_indicators": emr_columns,
        },
        "excluded_columns": {
            "health": sorted(HEALTH_EXCLUDE_COLUMNS),
            "insole": sorted(INSOLE_EXCLUDE_COLUMNS),
            "clinical_indicators": sorted(EMR_EXCLUDE_COLUMNS),
        },
        "categorical_encoding": {
            "health": HEALTH_CODEBOOK,
            "insole": INSOLE_CODEBOOK,
            "clinical_indicators": CLINICAL_CODEBOOK,
        },
        "row_meta_by_user": row_meta_by_user,
        "note": (
            "This generator writes the Linear Test_all_person MAT layout, including the shared "
            "EMR cycle. It supports both the legacy dataset folders and the generated Chinese "
            "folder layout; abnormal_group is treated as an explanatory source column and excluded "
            "from health features."
        ),
    }
    (output_dir / "conversion_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

def mat_payload(path: Path) -> dict[str, np.ndarray]:
    loaded = loadmat(path)
    return {key: loaded[key] for key in loaded if not key.startswith("__")}

def verify_mat_directory(output_dir: Path, reference_dir: Path) -> None:
    filenames = ["healthData.mat", "insoleData.mat", "EMRData.mat"]
    for filename in filenames:
        generated = mat_payload(output_dir / filename)
        reference = mat_payload(reference_dir / filename)

        generated_keys = sorted(generated)
        reference_keys = sorted(reference)
        if generated_keys != reference_keys:
            raise AssertionError(
                f"{filename} variable mismatch.\n"
                f"generated={generated_keys}\n"
                f"reference={reference_keys}"
            )

        for key in generated_keys:
            if generated[key].shape != reference[key].shape:
                raise AssertionError(
                    f"{filename}:{key} shape mismatch: generated={generated[key].shape}, "
                    f"reference={reference[key].shape}"
                )
            if not np.array_equal(generated[key], reference[key]):
                diff = np.argwhere(generated[key] != reference[key])[0]
                diff_tuple = tuple(int(index) for index in diff)
                raise AssertionError(
                    f"{filename}:{key} first mismatch at {diff_tuple}: "
                    f"generated={generated[key][diff_tuple]!r}, reference={reference[key][diff_tuple]!r}"
                )

def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    reference_dir = Path(args.reference_dir)
    versions = resolve_versions(dataset_root, parse_csv_list(args.versions))
    user_ids = parse_int_list(args.user_ids)
    if not user_ids:
        raise ValueError("--user-ids must contain at least one user id.")
    if user_ids != list(range(1, 11)) and not args.skip_verify:
        raise ValueError(
            "Reference verification is only valid for the legacy 1..10 all-person dataset. "
            "Use --skip-verify for newly generated target users."
        )

    layout = detect_dataset_layout(dataset_root / versions[0], user_ids[0])
    health_columns, insole_columns, emr_columns = read_feature_columns(
        dataset_root / versions[0],
        layout,
        user_ids[0],
    )
    rows_per_user = len(versions) * 10
    shared_emr = None
    if args.emr_layout == "legacy_shared_cycle":
        shared_emr = build_shared_emr_matrix(
            dataset_root=dataset_root,
            layout=layout,
            versions=versions,
            user_ids=user_ids,
            emr_columns=emr_columns,
        )
        if shared_emr.shape[0] != rows_per_user:
            raise ValueError(
                f"Shared EMR row count mismatch: got {shared_emr.shape[0]}, expected {rows_per_user}. "
                "Use --emr-layout per_user_repeated for target-only generated users."
            )
    elif args.emr_layout != "per_user_repeated":
        raise ValueError(f"Unsupported EMR layout: {args.emr_layout}")

    print("Generation plan")
    print(f"  dataset_root   : {dataset_root}")
    print(f"  dataset_layout : {layout.name}")
    print(f"  output_dir     : {output_dir}")
    print(f"  reference_dir  : {reference_dir}")
    print(f"  versions       : {versions}")
    print(f"  user_ids       : {user_ids}")
    print(f"  emr_layout     : {args.emr_layout}")
    print(f"  rows_per_user  : {rows_per_user}")
    print(f"  samples_per_day: {args.samples_per_day}")
    print(f"  feature_dims   : health={len(health_columns)}, insole={len(insole_columns)}, emr={len(emr_columns)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    health_vars: dict[str, np.ndarray] = {}
    insole_vars: dict[str, np.ndarray] = {}
    emr_vars: dict[str, np.ndarray] = {}
    row_meta_by_user: dict[str, list[dict[str, object]]] = {}

    for user_id in user_ids:
        health, insole, labels, row_meta = build_user_tensors(
            dataset_root=dataset_root,
            layout=layout,
            versions=versions,
            user_id=user_id,
            health_columns=health_columns,
            insole_columns=insole_columns,
            samples_per_day=args.samples_per_day,
        )
        if health.shape[0] != rows_per_user:
            raise ValueError(f"User {user_id} produced {health.shape[0]} rows, expected {rows_per_user}")

        health_vars[f"H_{user_id}_health"] = np.transpose(health, (0, 2, 1))
        health_vars[f"b_{user_id}"] = labels
        insole_vars[f"H_{user_id}_insole"] = np.transpose(insole, (0, 2, 1))
        if args.emr_layout == "legacy_shared_cycle":
            if shared_emr is None:
                raise RuntimeError("shared_emr was not built.")
            emr_vars[f"H_{user_id}_EMR"] = shared_emr
        else:
            emr_vars[f"H_{user_id}_EMR"] = build_per_user_repeated_emr_matrix(
                dataset_root=dataset_root,
                layout=layout,
                emr_columns=emr_columns,
                user_id=user_id,
                row_meta=row_meta,
            )
        row_meta_by_user[str(user_id)] = row_meta

        print(
            f"  user {user_id}: health={health_vars[f'H_{user_id}_health'].shape}, "
            f"insole={insole_vars[f'H_{user_id}_insole'].shape}, "
            f"emr={emr_vars[f'H_{user_id}_EMR'].shape}, labels={labels.shape}"
        )

    savemat(output_dir / "healthData.mat", health_vars, do_compression=True)
    savemat(output_dir / "insoleData.mat", insole_vars, do_compression=True)
    savemat(output_dir / "EMRData.mat", emr_vars, do_compression=True)
    write_manifest(
        output_dir=output_dir,
        dataset_root=dataset_root,
        layout=layout,
        versions=versions,
        user_ids=user_ids,
        health_columns=health_columns,
        insole_columns=insole_columns,
        emr_columns=emr_columns,
        emr_layout=args.emr_layout,
        samples_per_day=args.samples_per_day,
        row_meta_by_user=row_meta_by_user,
    )

    if not args.skip_verify:
        verify_mat_directory(output_dir, reference_dir)
        print("Verification passed: generated MAT payloads are exactly equal to the reference directory.")

    print(f"Saved: {output_dir / 'healthData.mat'}")
    print(f"Saved: {output_dir / 'insoleData.mat'}")
    print(f"Saved: {output_dir / 'EMRData.mat'}")
    print(f"Saved: {output_dir / 'conversion_manifest.json'}")

if __name__ == "__main__":
    main()
