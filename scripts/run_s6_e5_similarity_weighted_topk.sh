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
OUT_ROOT="${OUT_ROOT:-${STABLE_DIR}/results/s6_e5_similarity_weighted_topk}"
CACHE_ROOT="${CACHE_ROOT:-${OUT_ROOT}/source_response_cache}"
BASELINE_ROOT="${BASELINE_ROOT:-${STABLE_DIR}/results/s6_e2_target_full_recomputation}"

DEVICE="${DEVICE:-cpu}"
TORCH_DTYPE="${TORCH_DTYPE:-float64}"
STORAGE_DTYPE="${STORAGE_DTYPE:-float32}"
HISTORY_CHUNK_SIZE="${HISTORY_CHUNK_SIZE:-1024}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-0}"
VERSIONS="${VERSIONS:-v1,v2,v3,v4,v5,v6,v7,v8,v9,v10}"

SOURCE_USER_IDS="${SOURCE_USER_IDS:-1,2,3,4,5,6,7,8,9,10}"
TARGET_USER_IDS="${TARGET_USER_IDS:-11,12,13,14,15,16,17,18,19,20}"
METHOD2_K_VALUES="${METHOD2_K_VALUES:-1,2,3,4,5,6,7,8,9,10}"
REFERENCE_MODE="${REFERENCE_MODE:-first90}"
SIMILARITY_CHUNK_SIZE="${SIMILARITY_CHUNK_SIZE:-256}"
SIMILARITY_TIME_STEP_LIMIT="${SIMILARITY_TIME_STEP_LIMIT:-}"
TIME_STEP_LIMIT="${TIME_STEP_LIMIT:-}"
RUN_METHOD_SUMMARY="${RUN_METHOD_SUMMARY:-1}"
EXPECTED_TEST_DAYS="${EXPECTED_TEST_DAYS:-91,92,93,94,95,96,97,98,99,100}"

mat_complete() {
  [[ -f "$1/healthData.mat" && -f "$1/insoleData.mat" && -f "$1/EMRData.mat" ]]
}

json_list_matches() {
  local json_path="$1"
  local key_path="$2"
  local expected_csv="$3"
  [[ -f "${json_path}" ]] || return 1
  "${PYTHON_BIN}" -c '
import json
import sys

path, key_path, expected_csv = sys.argv[1:4]
expected = [int(value) for value in expected_csv.split(",") if value]
with open(path, encoding="utf-8") as handle:
    payload = json.load(handle)
node = payload
for key in key_path.split("."):
    node = node[int(key)] if key.isdigit() else node[key]
actual = [int(value) for value in node]
raise SystemExit(0 if actual == expected else 1)
' "${json_path}" "${key_path}" "${expected_csv}"
}

cache_complete() {
  local cache_root="$1"
  local user_csv="$2"
  local old_ifs="${IFS}"
  IFS=","
  read -r -a users <<< "${user_csv}"
  IFS="${old_ifs}"
  for user_id in "${users[@]}"; do
    user_id="${user_id//[[:space:]]/}"
    if [[ ! -f "${cache_root}/user$(printf "%02d" "${user_id}")/response_cache.npz" ]]; then
      return 1
    fi
    if ! json_list_matches \
      "${cache_root}/user$(printf "%02d" "${user_id}")/cache_manifest.json" \
      "days_1based" \
      "${EXPECTED_TEST_DAYS}"; then
      return 1
    fi
  done
  return 0
}

convert_source_mat() {
  if ! mat_complete "${SOURCE_MAT_DIR}"; then
    if [[ ! -d "${SOURCE_DATA_ROOT}" ]]; then
      echo "Missing source CSV data under SOURCE_DATA_ROOT=${SOURCE_DATA_ROOT}" >&2
      exit 1
    fi
    echo "Converting source CSV data from ${SOURCE_DATA_ROOT} into ${SOURCE_MAT_DIR}."
    "${PYTHON_BIN}" "${STABLE_DIR}/src/s6_e5_similarity_weighted_topk/generate_linear_test_mat.py" \
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
    "${PYTHON_BIN}" "${STABLE_DIR}/src/s6_e5_similarity_weighted_topk/generate_linear_test_mat.py" \
      --dataset-root "${TARGET_DATA_ROOT}" \
      --output-dir "${output_dir}" \
      --versions "${VERSIONS}" \
      --user-ids "${TARGET_USER_IDS}" \
      --skip-verify \
      --emr-layout per_user_repeated
  fi
}

if ! mat_complete "${DATA_DIR}"; then
  if [[ "${TARGET_EMR_MODE}" == "source_shared" ]]; then
    convert_source_mat
    convert_target_mat "${TARGET_MAT_DIR}"
    echo "Merging source and target MAT files into ${DATA_DIR} with source-shared target EMR."
    "${PYTHON_BIN}" "${STABLE_DIR}/src/s6_e5_similarity_weighted_topk/merge_source_target_mat.py" \
      --source-dir "${SOURCE_MAT_DIR}" \
      --target-dir "${TARGET_MAT_DIR}" \
      --output-dir "${DATA_DIR}" \
      --source-user-ids "${SOURCE_USER_IDS}" \
      --target-user-ids "${TARGET_USER_IDS}" \
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
  echo "Missing S6-E5 MAT input files under DATA_DIR=${DATA_DIR}" >&2
  echo "Expected: healthData.mat, insoleData.mat, EMRData.mat" >&2
  exit 1
fi

if [[ "${RUN_METHOD_SUMMARY}" == "1" && ! -f "${BASELINE_ROOT}/all_users_summary.json" ]]; then
  echo "Missing baseline summary under BASELINE_ROOT=${BASELINE_ROOT}" >&2
  echo "Run S6-E2 first or set RUN_METHOD_SUMMARY=0 to skip final method comparison." >&2
  exit 1
fi
if [[ "${RUN_METHOD_SUMMARY}" == "1" ]]; then
  if ! json_list_matches "${BASELINE_ROOT}/all_users_summary.json" "summaries.0.days_1based" "${EXPECTED_TEST_DAYS}"; then
    echo "Baseline summary under BASELINE_ROOT=${BASELINE_ROOT} does not use expected test days ${EXPECTED_TEST_DAYS}." >&2
    echo "Run S6-E2 with the v1-v9 train / v10 test split before S6-E5." >&2
    exit 1
  fi
fi

if ! cache_complete "${CACHE_ROOT}" "${SOURCE_USER_IDS}"; then
  echo "S6-E5 source response cache missing or not using expected test days ${EXPECTED_TEST_DAYS}; exporting it now."
  CACHE_COMMAND=(
    "${PYTHON_BIN}" "${STABLE_DIR}/src/s6_e5_similarity_weighted_topk/export_transfer_response_cache.py"
    --data-dir "${DATA_DIR}"
    --output-root "${CACHE_ROOT}"
    --baseline-root "${BASELINE_ROOT}"
    --user-ids "${SOURCE_USER_IDS}"
    --storage-dtype "${STORAGE_DTYPE}"
    --torch-dtype "${TORCH_DTYPE}"
    --device "${DEVICE}"
    --history-chunk-size "${HISTORY_CHUNK_SIZE}"
    --progress-interval "${PROGRESS_INTERVAL}"
  )
  if [[ -n "${TIME_STEP_LIMIT}" ]]; then
    CACHE_COMMAND+=(--time-step-limit "${TIME_STEP_LIMIT}")
  fi
  "${CACHE_COMMAND[@]}"
fi

TRANSFER_COMMAND=(
  "${PYTHON_BIN}" "${STABLE_DIR}/src/s6_e5_similarity_weighted_topk/run_transfer_learning_suite_paper_similarity.py"
  --data-dir "${DATA_DIR}"
  --baseline-root "${BASELINE_ROOT}"
  --response-cache-root "${CACHE_ROOT}"
  --output-root "${OUT_ROOT}"
  --target-user-ids "${TARGET_USER_IDS}"
  --candidate-user-ids "${SOURCE_USER_IDS}"
  --method2-k-values "${METHOD2_K_VALUES}"
  --storage-dtype "${STORAGE_DTYPE}"
  --reference-mode "${REFERENCE_MODE}"
  --similarity-chunk-size "${SIMILARITY_CHUNK_SIZE}"
)
if [[ -n "${SIMILARITY_TIME_STEP_LIMIT}" ]]; then
  TRANSFER_COMMAND+=(--similarity-time-step-limit "${SIMILARITY_TIME_STEP_LIMIT}")
fi
if [[ -n "${TIME_STEP_LIMIT}" ]]; then
  TRANSFER_COMMAND+=(--time-step-limit "${TIME_STEP_LIMIT}")
fi
"${TRANSFER_COMMAND[@]}"

if [[ "${RUN_METHOD_SUMMARY}" == "1" ]]; then
  "${PYTHON_BIN}" "${STABLE_DIR}/src/s6_e5_similarity_weighted_topk/summarize_transfer_learning_results.py" \
    --baseline-root "${BASELINE_ROOT}" \
    --transfer-root "${OUT_ROOT}" \
    --output-root "${OUT_ROOT}"
fi

"${PYTHON_BIN}" "${STABLE_DIR}/src/s6_e5_similarity_weighted_topk/plot_target_transfer_k_accuracy.py" \
  --k-sweep-csv "${OUT_ROOT}/method2_k_sweep.csv" \
  --pooled-metrics-csv "${OUT_ROOT}/method2_k_pooled_metrics.csv" \
  --output-path "${OUT_ROOT}/target_transfer_k_accuracy.png" \
  --mean-csv "${OUT_ROOT}/target_transfer_k_accuracy_mean.csv"

"${PYTHON_BIN}" "${STABLE_DIR}/src/s6_e5_similarity_weighted_topk/plot_target_transfer_k_boxplot.py" \
  --k-sweep-csv "${OUT_ROOT}/method2_k_sweep.csv" \
  --output-path "${OUT_ROOT}/target_transfer_k_accuracy_boxplot.png" \
  --stats-csv "${OUT_ROOT}/target_transfer_k_accuracy_boxplot_stats.csv"

"${PYTHON_BIN}" "${STABLE_DIR}/src/s6_e5_similarity_weighted_topk/summarize_topk_transfer.py" \
  --result-root "${OUT_ROOT}" \
  --out-root "${OUT_ROOT}"
