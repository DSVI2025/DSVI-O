#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scipy.io import loadmat, savemat

def parse_ints(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one user id.")
    return values

def load_vars(mat_path: Path, variable_names: list[str]) -> dict[str, object]:
    missing: list[str] = []
    loaded = loadmat(mat_path, variable_names=variable_names)
    output: dict[str, object] = {}
    for name in variable_names:
        if name not in loaded:
            missing.append(name)
        else:
            output[name] = loaded[name]
    if missing:
        raise KeyError(f"Missing variables in {mat_path}: {missing}")
    return output

def merge_mat_file(
    *,
    source_dir: Path,
    target_dir: Path,
    output_dir: Path,
    filename: str,
    source_vars: list[str],
    target_vars: list[str],
) -> dict[str, object]:
    payload: dict[str, object] = {}
    payload.update(load_vars(source_dir / filename, source_vars))
    payload.update(load_vars(target_dir / filename, target_vars))
    savemat(output_dir / filename, payload, do_compression=True)
    return {
        "filename": filename,
        "source_variables": source_vars,
        "target_variables": target_vars,
        "num_variables": len(payload),
    }

def merge_emr_file(
    *,
    source_dir: Path,
    target_dir: Path,
    output_dir: Path,
    source_user_ids: list[int],
    target_user_ids: list[int],
    target_emr_mode: str,
    source_shared_user_id: int,
) -> dict[str, object]:
    filename = "EMRData.mat"
    source_vars = [f"H_{user_id}_EMR" for user_id in source_user_ids]
    target_vars = [f"H_{user_id}_EMR" for user_id in target_user_ids]
    payload = load_vars(source_dir / filename, source_vars)

    if target_emr_mode == "target":
        payload.update(load_vars(target_dir / filename, target_vars))
    elif target_emr_mode == "source_shared":
        shared_name = f"H_{source_shared_user_id}_EMR"
        if shared_name not in payload:
            payload.update(load_vars(source_dir / filename, [shared_name]))
        shared_value = payload[shared_name]
        for target_var in target_vars:
            payload[target_var] = shared_value
    else:
        raise ValueError(f"Unsupported target_emr_mode: {target_emr_mode}")

    savemat(output_dir / filename, payload, do_compression=True)
    return {
        "filename": filename,
        "source_variables": source_vars,
        "target_variables": target_vars,
        "target_emr_mode": target_emr_mode,
        "source_shared_user_id": source_shared_user_id,
        "num_variables": len(payload),
    }

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge legacy source-user MAT variables with newly generated target-user MAT variables."
    )
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-user-ids", default="1,2,3,4,5,6,7,8,9,10")
    parser.add_argument("--target-user-ids", default="11,12,13")
    parser.add_argument(
        "--target-emr-mode",
        choices=["target", "source_shared"],
        default="target",
        help=(
            "target keeps target EMR variables from target-dir. "
            "source_shared gives each target the legacy shared source EMR matrix for compatibility "
            "with the existing Round31 source cache."
        ),
    )
    parser.add_argument(
        "--source-shared-user-id",
        type=int,
        default=1,
        help="Source user whose H_<id>_EMR matrix is used when --target-emr-mode source_shared.",
    )
    args = parser.parse_args()

    source_user_ids = parse_ints(args.source_user_ids)
    target_user_ids = parse_ints(args.target_user_ids)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest_items = []
    manifest_items.append(
        merge_mat_file(
            source_dir=args.source_dir,
            target_dir=args.target_dir,
            output_dir=args.output_dir,
            filename="healthData.mat",
            source_vars=[
                name
                for user_id in source_user_ids
                for name in (f"H_{user_id}_health", f"b_{user_id}")
            ],
            target_vars=[
                name
                for user_id in target_user_ids
                for name in (f"H_{user_id}_health", f"b_{user_id}")
            ],
        )
    )
    manifest_items.append(
        merge_mat_file(
            source_dir=args.source_dir,
            target_dir=args.target_dir,
            output_dir=args.output_dir,
            filename="insoleData.mat",
            source_vars=[f"H_{user_id}_insole" for user_id in source_user_ids],
            target_vars=[f"H_{user_id}_insole" for user_id in target_user_ids],
        )
    )
    manifest_items.append(
        merge_emr_file(
            source_dir=args.source_dir,
            target_dir=args.target_dir,
            output_dir=args.output_dir,
            source_user_ids=source_user_ids,
            target_user_ids=target_user_ids,
            target_emr_mode=args.target_emr_mode,
            source_shared_user_id=args.source_shared_user_id,
        )
    )

    manifest = {
        "source_dir": str(args.source_dir),
        "target_dir": str(args.target_dir),
        "output_dir": str(args.output_dir),
        "source_user_ids": source_user_ids,
        "target_user_ids": target_user_ids,
        "target_emr_mode": args.target_emr_mode,
        "source_shared_user_id": args.source_shared_user_id,
        "items": manifest_items,
    }
    (args.output_dir / "merge_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    for item in manifest_items:
        print(
            f"{item['filename']}: source={len(item['source_variables'])} "
            f"target={len(item['target_variables'])} total={item['num_variables']}"
        )
    print(f"Saved merged MAT directory: {args.output_dir}")

if __name__ == "__main__":
    main()
