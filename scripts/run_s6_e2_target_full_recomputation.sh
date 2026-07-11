#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STABLE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

TARGET_EMR_MODE="${TARGET_EMR_MODE:-source_shared}"
MAT_ROOT="${MAT_ROOT:-${STABLE_DIR}/data/mat}"
SOURCE_MAT_DIR="${SOURCE_MAT_DIR:-${MAT_ROOT}/source_legacy_shared_emr}"
TARGET_MAT_DIR="${TARGET_MAT_DIR:-${MAT_ROOT}/target}"
if [[ -z "${DATA_DIR:-}" ]]; then
  if [[ "${TARGET_EMR_MODE}" == "source_shared" ]]; then
    DATA_DIR="${MAT_ROOT}/source_target_shared_emr"
  else
    DATA_DIR="${TARGET_MAT_DIR}"
  fi
fi
SOURCE_DATA_ROOT="${SOURCE_DATA_ROOT:-${STABLE_DIR}/data/source}"
TARGET_DATA_ROOT="${TARGET_DATA_ROOT:-${STABLE_DIR}/data/target}"
OUT_ROOT="${OUT_ROOT:-${STABLE_DIR}/results/s6_e2_target_full_recomputation}"
DEVICE="${DEVICE:-cpu}"
TORCH_DTYPE="${TORCH_DTYPE:-float64}"
STORAGE_DTYPE="${STORAGE_DTYPE:-float32}"
TARGET_DAY_BATCH_SIZE="${TARGET_DAY_BATCH_SIZE:-10}"
HISTORY_CHUNK_SIZE="${HISTORY_CHUNK_SIZE:-1024}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-0}"
VERSIONS="${VERSIONS:-v1,v2,v3,v4,v5,v6,v7,v8,v9,v10}"

SOURCE_USER_IDS="${SOURCE_USER_IDS:-1,2,3,4,5,6,7,8,9,10}"
USER_IDS="${USER_IDS:-11,12,13,14,15,16,17,18,19,20}"

mat_complete() {
  [[ -f "$1/healthData.mat" && -f "$1/insoleData.mat" && -f "$1/EMRData.mat" ]]
}

convert_source_mat() {
  if ! mat_complete "${SOURCE_MAT_DIR}"; then
    if [[ ! -d "${SOURCE_DATA_ROOT}" ]]; then
      echo "Missing source CSV data under SOURCE_DATA_ROOT=${SOURCE_DATA_ROOT}" >&2
      exit 1
    fi
    echo "Converting source CSV data from ${SOURCE_DATA_ROOT} into ${SOURCE_MAT_DIR}."
    "${PYTHON_BIN}" "${STABLE_DIR}/src/s6_e2_target_full_recomputation/generate_linear_test_mat.py" \
      --dataset-root "${SOURCE_DATA_ROOT}" \
      --output-dir "${SOURCE_MAT_DIR}" \
      --versions "${VERSIONS}" \
      --user-ids "${SOURCE_USER_IDS}" \
      --skip-verify \
      --emr-layout legacy_shared_cycle
  fi
}

convert_target_mat() {
  local output_dir="$1"
  if ! mat_complete "${output_dir}"; then
    if [[ ! -d "${TARGET_DATA_ROOT}" ]]; then
      echo "Missing target CSV data under TARGET_DATA_ROOT=${TARGET_DATA_ROOT}" >&2
      exit 1
    fi
    echo "Converting target CSV data from ${TARGET_DATA_ROOT} into ${output_dir}."
    "${PYTHON_BIN}" "${STABLE_DIR}/src/s6_e2_target_full_recomputation/generate_linear_test_mat.py" \
      --dataset-root "${TARGET_DATA_ROOT}" \
      --output-dir "${output_dir}" \
      --versions "${VERSIONS}" \
      --user-ids "${USER_IDS}" \
      --skip-verify \
      --emr-layout per_user_repeated
  fi
}

if ! mat_complete "${DATA_DIR}"; then
  if [[ "${TARGET_EMR_MODE}" == "source_shared" ]]; then
    convert_source_mat
    convert_target_mat "${TARGET_MAT_DIR}"
    echo "Merging source and target MAT files into ${DATA_DIR} with source-shared target EMR."
    "${PYTHON_BIN}" "${STABLE_DIR}/src/s6_e2_target_full_recomputation/merge_source_target_mat.py" \
      --source-dir "${SOURCE_MAT_DIR}" \
      --target-dir "${TARGET_MAT_DIR}" \
      --output-dir "${DATA_DIR}" \
      --source-user-ids "${SOURCE_USER_IDS}" \
      --target-user-ids "${USER_IDS}" \
      --target-emr-mode source_shared \
      --source-shared-user-id 1
  elif [[ "${TARGET_EMR_MODE}" == "target" ]]; then
    convert_target_mat "${DATA_DIR}"
  else
    echo "Unsupported TARGET_EMR_MODE=${TARGET_EMR_MODE}; expected source_shared or target." >&2
    exit 1
  fi
fi

if ! mat_complete "${DATA_DIR}"; then
  echo "Missing S6-E2 MAT input files under DATA_DIR=${DATA_DIR}" >&2
  echo "Expected: healthData.mat, insoleData.mat, EMRData.mat" >&2
  echo "Set DATA_DIR to an existing MAT directory or set SOURCE_DATA_ROOT/TARGET_DATA_ROOT to the committed CSV data." >&2
  exit 1
fi

"${PYTHON_BIN}" "${STABLE_DIR}/src/s6_e2_target_full_recomputation/run_old_matlab_equivalent.py" \
  --data-dir "${DATA_DIR}" \
  --out-root "${OUT_ROOT}" \
  --user-ids "${USER_IDS}" \
  --noise-preset none \
  --reference-sampling-mode sampled \
  --reference-sample-size 50 \
  --reference-num-trials 20 \
  --reference-sampling-seed 20260423 \
  --history-window 12 \
  --storage-dtype "${STORAGE_DTYPE}" \
  --device "${DEVICE}" \
  --torch-dtype "${TORCH_DTYPE}" \
  --target-day-batch-size "${TARGET_DAY_BATCH_SIZE}" \
  --history-chunk-size "${HISTORY_CHUNK_SIZE}" \
  --progress-interval "${PROGRESS_INTERVAL}" \
  --params-hat-rho 0.7 \
  --params-hat-lambda 0.305 \
  --params-hat-alpha-mix 0.75,0.58,0.16

"${PYTHON_BIN}" "${STABLE_DIR}/src/s6_e2_target_full_recomputation/summarize_target_full_recomputation.py" \
  --result-root "${OUT_ROOT}" \
  --out-root "${OUT_ROOT}"
