#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STABLE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

TARGET_EMR_MODE="${TARGET_EMR_MODE:-source_shared}"
MAT_ROOT="${MAT_ROOT:-${STABLE_DIR}/data/mat}"
if [[ -z "${DATA_DIR:-}" ]]; then
  if [[ "${TARGET_EMR_MODE}" == "source_shared" ]]; then
    DATA_DIR="${MAT_ROOT}/source_target_shared_emr"
  else
    DATA_DIR="${MAT_ROOT}/target"
  fi
fi

S6_E5_ROOT="${S6_E5_ROOT:-${STABLE_DIR}/results/s6_e5_similarity_weighted_topk}"
RESPONSE_CACHE_ROOT="${RESPONSE_CACHE_ROOT:-${S6_E5_ROOT}/source_response_cache}"
PAIRWISE_JSON="${PAIRWISE_JSON:-${S6_E5_ROOT}/pairwise_similarity.json}"
OUT_ROOT="${OUT_ROOT:-${STABLE_DIR}/results/s6_e7_delayed_response_reuse}"

K="${K:-8}"
REFRESH_INTERVALS="${REFRESH_INTERVALS:-${LAGS:-0,6,12,60,120,360,720,2880,8640}}"
TARGET_USER_IDS="${TARGET_USER_IDS:-11,12,13,14,15,16,17,18,19,20}"
STORAGE_DTYPE="${STORAGE_DTYPE:-float32}"
RUN_DEPENDENCIES_IF_MISSING="${RUN_DEPENDENCIES_IF_MISSING:-0}"
MAKE_PLOTS="${MAKE_PLOTS:-1}"

if [[ ! -f "${PAIRWISE_JSON}" || ! -d "${RESPONSE_CACHE_ROOT}" ]]; then
  if [[ "${RUN_DEPENDENCIES_IF_MISSING}" == "1" ]]; then
    echo "Missing S6-E5 transfer artifacts; running S6-E5 first."
    "${STABLE_DIR}/scripts/run_s6_e5_similarity_weighted_topk.sh"
  else
    echo "Missing S6-E5 artifacts." >&2
    echo "Expected PAIRWISE_JSON=${PAIRWISE_JSON}" >&2
    echo "Expected RESPONSE_CACHE_ROOT=${RESPONSE_CACHE_ROOT}" >&2
    echo "Run S6-E5 first or set RUN_DEPENDENCIES_IF_MISSING=1." >&2
    exit 1
  fi
fi

if [[ ! -f "${DATA_DIR}/healthData.mat" || ! -f "${DATA_DIR}/insoleData.mat" || ! -f "${DATA_DIR}/EMRData.mat" ]]; then
  echo "Missing MAT input files under DATA_DIR=${DATA_DIR}" >&2
  echo "Run S6-E5/S6-E2 data preparation first." >&2
  exit 1
fi

"${PYTHON_BIN}" "${STABLE_DIR}/src/s6_e7_delayed_response_reuse/run_lagged_response_transfer.py" \
  --data-dir "${DATA_DIR}" \
  --response-cache-root "${RESPONSE_CACHE_ROOT}" \
  --pairwise-json "${PAIRWISE_JSON}" \
  --output-dir "${OUT_ROOT}" \
  --k "${K}" \
  --refresh-intervals "${REFRESH_INTERVALS}" \
  --user-ids "${TARGET_USER_IDS}" \
  --storage-dtype "${STORAGE_DTYPE}"

if [[ "${MAKE_PLOTS}" == "1" ]]; then
  "${PYTHON_BIN}" "${STABLE_DIR}/src/s6_e7_delayed_response_reuse/plot_temporal_robustness_summary.py" \
    --metrics-csv "${OUT_ROOT}/lagged_response_transfer_metrics.csv" \
    --accuracy-output "${OUT_ROOT}/application_temporal_lag_accuracy.png" \
    --tradeoff-output "${OUT_ROOT}/application_response_update_savings_accuracy.png" \
    --stats-output "${OUT_ROOT}/temporal_robustness_stats.csv"

  "${PYTHON_BIN}" "${STABLE_DIR}/src/s6_e7_delayed_response_reuse/plot_response_update_savings_accuracy.py" \
    --summary-csv "${OUT_ROOT}/lagged_response_transfer_summary.csv" \
    --output-png "${OUT_ROOT}/response_update_savings_accuracy.png" \
    --output-csv "${OUT_ROOT}/response_update_savings_accuracy.csv"
fi

"${PYTHON_BIN}" "${STABLE_DIR}/src/s6_e7_delayed_response_reuse/summarize_delayed_response_reuse.py" \
  --summary-csv "${OUT_ROOT}/lagged_response_transfer_summary.csv" \
  --output-root "${OUT_ROOT}"
