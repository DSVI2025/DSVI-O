#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STABLE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

COMPARISON_CSV="${COMPARISON_CSV:-${STABLE_DIR}/results/s6_e5_similarity_weighted_topk/final_method_comparison.csv}"
OUT_ROOT="${OUT_ROOT:-${STABLE_DIR}/results/s6_e6_transfer_latency}"
POINTS_PER_BATCH="${POINTS_PER_BATCH:-172800}"
RUN_DEPENDENCIES_IF_MISSING="${RUN_DEPENDENCIES_IF_MISSING:-0}"

if [[ ! -f "${COMPARISON_CSV}" ]]; then
  if [[ "${RUN_DEPENDENCIES_IF_MISSING}" == "1" ]]; then
    echo "Missing ${COMPARISON_CSV}; running S6-E5 first."
    "${STABLE_DIR}/scripts/run_s6_e5_similarity_weighted_topk.sh"
  else
    echo "Missing comparison CSV: ${COMPARISON_CSV}" >&2
    echo "Run S6-E5 first, set COMPARISON_CSV, or set RUN_DEPENDENCIES_IF_MISSING=1." >&2
    exit 1
  fi
fi

"${PYTHON_BIN}" "${STABLE_DIR}/src/s6_e6_transfer_latency/summarize_transfer_method_latency.py" \
  --comparison-csv "${COMPARISON_CSV}" \
  --output-root "${OUT_ROOT}" \
  --points-per-batch "${POINTS_PER_BATCH}"
