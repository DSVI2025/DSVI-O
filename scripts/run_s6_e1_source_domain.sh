#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STABLE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

DATA_DIR="${DATA_DIR:-${STABLE_DIR}/data/mat/source}"
SOURCE_DATA_ROOT="${SOURCE_DATA_ROOT:-${STABLE_DIR}/data/source}"
OUT_ROOT="${OUT_ROOT:-${STABLE_DIR}/results/s6_e1_source_domain}"
DEVICE="${DEVICE:-cpu}"
TORCH_DTYPE="${TORCH_DTYPE:-float64}"
STORAGE_DTYPE="${STORAGE_DTYPE:-float32}"
TARGET_DAY_BATCH_SIZE="${TARGET_DAY_BATCH_SIZE:-10}"
HISTORY_CHUNK_SIZE="${HISTORY_CHUNK_SIZE:-1024}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-0}"
VERSIONS="${VERSIONS:-v1,v2,v3,v4,v5,v6,v7,v8,v9,v10}"

USER_IDS="${USER_IDS:-1,2,3,4,5,6,7,8,9,10}"
REFERENCE_DAYS="${REFERENCE_DAYS:-1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90}"
TEST_DAYS="${TEST_DAYS:-91,92,93,94,95,96,97,98,99,100}"

if [[ ! -f "${DATA_DIR}/healthData.mat" || ! -f "${DATA_DIR}/insoleData.mat" || ! -f "${DATA_DIR}/EMRData.mat" ]]; then
  if [[ -d "${SOURCE_DATA_ROOT}" ]]; then
    echo "S6-E1 MAT input not found under DATA_DIR=${DATA_DIR}; converting source CSV data from ${SOURCE_DATA_ROOT}."
    "${PYTHON_BIN}" "${STABLE_DIR}/src/s6_e1_source_domain/generate_linear_test_mat.py" \
      --dataset-root "${SOURCE_DATA_ROOT}" \
      --output-dir "${DATA_DIR}" \
      --versions "${VERSIONS}" \
      --user-ids "${USER_IDS}" \
      --skip-verify \
      --emr-layout per_user_repeated
  fi
fi

if [[ ! -f "${DATA_DIR}/healthData.mat" || ! -f "${DATA_DIR}/insoleData.mat" || ! -f "${DATA_DIR}/EMRData.mat" ]]; then
  echo "Missing S6-E1 MAT input files under DATA_DIR=${DATA_DIR}" >&2
  echo "Expected: healthData.mat, insoleData.mat, EMRData.mat" >&2
  echo "Set DATA_DIR to an existing MAT directory or set SOURCE_DATA_ROOT to generated source CSV data." >&2
  exit 1
fi

"${PYTHON_BIN}" "${STABLE_DIR}/src/s6_e1_source_domain/run_source_domain_v10_test.py" \
  --code-root "${STABLE_DIR}/src/s6_e1_source_domain" \
  --data-dir "${DATA_DIR}" \
  --out-root "${OUT_ROOT}" \
  --user-ids "${USER_IDS}" \
  --reference-days "${REFERENCE_DAYS}" \
  --test-days "${TEST_DAYS}" \
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
