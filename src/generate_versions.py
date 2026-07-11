#!/usr/bin/env python3
"""Generate the Section 6 source/target synthetic cohorts.

This module keeps the original DSVI-O data-generation logic self-contained
under src. It first writes user profiles and EMR records, then generates
10-day wearable and insole time series for each version.
"""

from __future__ import annotations

import argparse
import shutil
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import pandas as pd

from generate_timeseries import process_user_data
from health_gen.user_generator import UserGenerator

DEFAULT_START_VERSION = 1
DEFAULT_NUM_VERSIONS = 10
DEFAULT_NUM_USERS = 10
DEFAULT_SOURCE_USER_SEED = 53
DEFAULT_TARGET_USER_SEED = 73
DEFAULT_TARGET_USER_ID_OFFSET = 10

def _stable_dir() -> Path:
    return Path(__file__).resolve().parents[1]

def _version_name(version_index: int, version_prefix: str) -> str:
    return f"{version_prefix}{version_index}"

def _version_indices(start_version: int, num_versions: int) -> range:
    if start_version < 1:
        raise ValueError("--start-version must be >= 1")
    if num_versions < 1:
        raise ValueError("--num-versions must be >= 1")
    return range(start_version, start_version + num_versions)

def _base_data_dir(output_root: Path, version: str) -> Path:
    return output_root / version / "generated_data"

def generate_profiles_and_emr(
    output_root: Path,
    version: str,
    num_users: int,
    user_seed: int,
    user_id_offset: int,
    overwrite: bool,
) -> pd.DataFrame:
    version_dir = output_root / version
    if overwrite and version_dir.exists():
        shutil.rmtree(version_dir)

    base_dir = _base_data_dir(output_root, version)
    emr_dir = base_dir / "electronic_medical_records"
    emr_dir.mkdir(parents=True, exist_ok=True)

    generator = UserGenerator(seed=user_seed)
    user_df = generator.generate_user_profile(num_users).copy()
    user_df["user_id"] = np.arange(1, len(user_df) + 1)

    medical_df = generator.generate_medical_record(user_df)

    if user_id_offset:
        id_map = {
            int(old_id): int(old_id) + user_id_offset
            for old_id in user_df["user_id"].astype(int).tolist()
        }
        user_df["user_id"] = user_df["user_id"].astype(int).map(id_map)
        medical_df["PatientID"] = medical_df["PatientID"].astype(int).map(id_map)

    user_columns = ["user_id", "age", "gender", "height_cm", "weight_kg", "BMI"]
    user_df[user_columns].to_csv(base_dir / "user_profiles.csv", index=False)
    medical_df.to_csv(base_dir / "medical_records.csv", index=False)

    for row in medical_df.itertuples(index=False):
        patient_id = int(row.PatientID)
        patient_df = medical_df[medical_df["PatientID"] == patient_id]
        patient_df.to_csv(emr_dir / f"{patient_id}_EMR.csv", index=False)

    return user_df[user_columns]

def generate_timeseries(
    output_root: Path,
    version: str,
    base_seed: int,
    processes: int,
) -> list[dict]:
    base_dir = _base_data_dir(output_root, version)
    user_df = pd.read_csv(base_dir / "user_profiles.csv")
    medical_df = pd.read_csv(base_dir / "medical_records.csv")

    user_data_list = []
    for _, row in user_df.iterrows():
        user_id = int(row["user_id"])
        user_profile = row.to_dict()
        medical_record = medical_df[medical_df["PatientID"] == user_id].iloc[0].to_dict()
        user_data_list.append((user_id, user_profile, medical_record))

    worker = partial(
        process_user_data,
        workdir=str(output_root),
        version=version,
        base_seed=base_seed,
        plot=False,
    )

    if processes == 1:
        return [worker(user_data) for user_data in user_data_list]

    with Pool(processes=processes) as pool:
        return pool.map(worker, user_data_list)

def generate_versions(
    kind: str,
    output_root: Path,
    start_version: int,
    num_versions: int,
    version_prefix: str,
    num_users: int,
    user_seed: int,
    user_id_offset: int,
    processes: int,
    overwrite: bool,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"kind={kind}")
    print(f"output_root={output_root}")
    print(f"start_version={start_version}")
    print(f"num_versions={num_versions}")
    print(f"num_users={num_users}")
    print(f"user_seed={user_seed}")
    print(f"user_id_offset={user_id_offset}")
    print(f"processes={processes}")
    print(f"overwrite={overwrite}")

    for version_index in _version_indices(start_version, num_versions):
        version = _version_name(version_index, version_prefix)
        print()
        print(f"=== Generating {kind} {version} ===")

        users = generate_profiles_and_emr(
            output_root=output_root,
            version=version,
            num_users=num_users,
            user_seed=user_seed,
            user_id_offset=user_id_offset,
            overwrite=overwrite,
        )
        user_ids = users["user_id"].astype(int).tolist()
        print(f"user_ids={user_ids[0]}-{user_ids[-1]}")

        results = generate_timeseries(
            output_root=output_root,
            version=version,
            base_seed=version_index,
            processes=processes,
        )

        total_health_rows = sum(result["daily_shape"][0] for result in results)
        total_insole_rows = sum(result["insole_shape"][0] for result in results)
        print(f"health_rows={total_health_rows}")
        print(f"insole_rows={total_insole_rows}")

    last_version = start_version + num_versions - 1
    print()
    print(f"Generated {kind} versions: {version_prefix}{start_version}-{version_prefix}{last_version}")
    print(f"Output root: {output_root}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["source", "target"], required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--start-version", type=int, default=DEFAULT_START_VERSION)
    parser.add_argument("--num-versions", type=int, default=DEFAULT_NUM_VERSIONS)
    parser.add_argument("--version-prefix", default="v")
    parser.add_argument("--num-users", type=int, default=DEFAULT_NUM_USERS)
    parser.add_argument("--user-seed", type=int)
    parser.add_argument("--user-id-offset", type=int)
    parser.add_argument("--processes", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()

def main() -> None:
    args = parse_args()

    output_root = args.output_root or (_stable_dir() / "data" / args.kind)
    user_seed = args.user_seed
    if user_seed is None:
        user_seed = DEFAULT_TARGET_USER_SEED if args.kind == "target" else DEFAULT_SOURCE_USER_SEED

    user_id_offset = args.user_id_offset
    if user_id_offset is None:
        user_id_offset = DEFAULT_TARGET_USER_ID_OFFSET if args.kind == "target" else 0

    processes = args.processes or min(cpu_count(), args.num_users)

    generate_versions(
        kind=args.kind,
        output_root=output_root,
        start_version=args.start_version,
        num_versions=args.num_versions,
        version_prefix=args.version_prefix,
        num_users=args.num_users,
        user_seed=user_seed,
        user_id_offset=user_id_offset,
        processes=processes,
        overwrite=args.overwrite,
    )

if __name__ == "__main__":
    main()
