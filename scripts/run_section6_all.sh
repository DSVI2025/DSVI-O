#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STABLE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

export PYTHON_BIN="${PYTHON_BIN:-python3}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${STABLE_DIR}/.mplconfig}"
mkdir -p "${MPLCONFIGDIR}"

if [[ "${CLEAN:-1}" == "1" ]]; then
  rm -rf \
    "${STABLE_DIR}/data" \
    "${STABLE_DIR}/results" \
    "${STABLE_DIR}/.mplconfig"
  mkdir -p "${MPLCONFIGDIR}"
fi

export OVERWRITE="${OVERWRITE:-1}"
export PROCESSES="${PROCESSES:-10}"
export TARGET_EMR_MODE="${TARGET_EMR_MODE:-source_shared}"
export DEVICE="${DEVICE:-cuda}"
export TORCH_DTYPE="${TORCH_DTYPE:-float64}"
export STORAGE_DTYPE="${STORAGE_DTYPE:-float32}"
export TARGET_DAY_BATCH_SIZE="${TARGET_DAY_BATCH_SIZE:-10}"
export HISTORY_CHUNK_SIZE="${HISTORY_CHUNK_SIZE:-1024}"
export PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-0}"
export SIMILARITY_CHUNK_SIZE="${SIMILARITY_CHUNK_SIZE:-256}"
export MAKE_PLOTS="${MAKE_PLOTS:-1}"

echo "[section6] S6-E0 data generation"
"${STABLE_DIR}/scripts/generate_section6_data.sh"

echo "[section6] S6-E1 source-domain benchmark"
"${STABLE_DIR}/scripts/run_s6_e1_source_domain.sh"

echo "[section6] S6-E2 target full recomputation"
"${STABLE_DIR}/scripts/run_s6_e2_target_full_recomputation.sh"

echo "[section6] S6-E2a target response density"
"${STABLE_DIR}/scripts/run_s6_e2_target_response_density.sh"

echo "[section6] S6-E3 noise robustness"
"${STABLE_DIR}/scripts/run_s6_e3_noise_robustness.sh"

echo "[section6] S6-E4 source-target similarity"
"${STABLE_DIR}/scripts/run_s6_e4_source_target_similarity.sh"

echo "[section6] S6-E5 similarity-weighted transfer"
"${STABLE_DIR}/scripts/run_s6_e5_similarity_weighted_topk.sh"

echo "[section6] S6-E6 transfer latency"
"${STABLE_DIR}/scripts/run_s6_e6_transfer_latency.sh"

echo "[section6] S6-E7 delayed response reuse"
"${STABLE_DIR}/scripts/run_s6_e7_delayed_response_reuse.sh"

echo "[section6] complete"
