#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STABLE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHON_BIN

export MPLCONFIGDIR="${MPLCONFIGDIR:-${STABLE_DIR}/.mplconfig}"
mkdir -p "${MPLCONFIGDIR}"

if [[ "${CLEAN:-1}" == "1" ]]; then
  rm -rf \
    "${STABLE_DIR}/data" \
    "${STABLE_DIR}/results/s6_e2_target_full_recomputation" \
    "${STABLE_DIR}/results/s6_e5_similarity_weighted_topk"
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

"${STABLE_DIR}/scripts/generate_section6_data.sh"
"${STABLE_DIR}/scripts/run_s6_e2_target_full_recomputation.sh"
"${STABLE_DIR}/scripts/run_s6_e5_similarity_weighted_topk.sh"
