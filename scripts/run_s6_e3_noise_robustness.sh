#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STABLE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

DATA_DIR="${DATA_DIR:-${STABLE_DIR}/data/mat/source_legacy_shared_emr}"
SOURCE_DATA_ROOT="${SOURCE_DATA_ROOT:-${STABLE_DIR}/data/source}"
OUT_ROOT="${OUT_ROOT:-${STABLE_DIR}/results/s6_e3_noise_robustness}"
DEVICE="${DEVICE:-cpu}"
TORCH_DTYPE="${TORCH_DTYPE:-float64}"
STORAGE_DTYPE="${STORAGE_DTYPE:-float32}"
TARGET_DAY_BATCH_SIZE="${TARGET_DAY_BATCH_SIZE:-10}"
HISTORY_CHUNK_SIZE="${HISTORY_CHUNK_SIZE:-1024}"
MAX_PARALLEL="${MAX_PARALLEL:-1}"
VERSIONS="${VERSIONS:-v1,v2,v3,v4,v5,v6,v7,v8,v9,v10}"

USER_IDS="${USER_IDS:-1}"
SOURCE_USER_IDS="${SOURCE_USER_IDS:-1,2,3,4,5,6,7,8,9,10}"
NOISE_COMPONENTS="${NOISE_COMPONENTS:-additive,drift,multiplicative,cumulative,laplace,impulse,mixed}"
STRENGTHS="${STRENGTHS:-0.5,1.0,1.5,2.0,5.0,10.0,20.0}"
NOISE_APPLY_TO="${NOISE_APPLY_TO:-all}"

mat_complete() {
  [[ -f "$1/healthData.mat" && -f "$1/insoleData.mat" && -f "$1/EMRData.mat" ]]
}

if ! mat_complete "${DATA_DIR}"; then
  if [[ ! -d "${SOURCE_DATA_ROOT}" ]]; then
    echo "Missing source CSV data under SOURCE_DATA_ROOT=${SOURCE_DATA_ROOT}" >&2
    exit 1
  fi
  echo "Converting source CSV data from ${SOURCE_DATA_ROOT} into ${DATA_DIR}."
  "${PYTHON_BIN}" "${STABLE_DIR}/src/s6_e3_noise_robustness/generate_linear_test_mat.py" \
    --dataset-root "${SOURCE_DATA_ROOT}" \
    --output-dir "${DATA_DIR}" \
    --versions "${VERSIONS}" \
    --user-ids "${SOURCE_USER_IDS}" \
    --skip-verify \
    --emr-layout legacy_shared_cycle
fi

if ! mat_complete "${DATA_DIR}"; then
  echo "Missing S6-E3 MAT input files under DATA_DIR=${DATA_DIR}" >&2
  echo "Expected: healthData.mat, insoleData.mat, EMRData.mat" >&2
  echo "Set DATA_DIR to an existing MAT directory or set SOURCE_DATA_ROOT to generated source CSV data." >&2
  exit 1
fi

"${PYTHON_BIN}" "${STABLE_DIR}/src/s6_e3_noise_robustness/run_noise_sweep_suite.py" \
  --data-dir "${DATA_DIR}" \
  --output-root "${OUT_ROOT}" \
  --user-ids "${USER_IDS}" \
  --noise-components "${NOISE_COMPONENTS}" \
  --strengths "${STRENGTHS}" \
  --num-repeats 1 \
  --noise-seed 20260423 \
  --storage-dtype "${STORAGE_DTYPE}" \
  --device "${DEVICE}" \
  --torch-dtype "${TORCH_DTYPE}" \
  --target-day-batch-size "${TARGET_DAY_BATCH_SIZE}" \
  --history-chunk-size "${HISTORY_CHUNK_SIZE}" \
  --noise-apply-to "${NOISE_APPLY_TO}" \
  --reference-sampling-mode sampled \
  --reference-sample-size 50 \
  --reference-num-trials 20 \
  --reference-sampling-seed 20260423 \
  --history-window 12 \
  --params-hat-rho 0.7 \
  --params-hat-lambda 0.305 \
  --params-hat-alpha-mix 0.75,0.58,0.16 \
  --max-parallel "${MAX_PARALLEL}"

"${PYTHON_BIN}" "${STABLE_DIR}/src/s6_e3_noise_robustness/summarize_noise_robustness.py" \
  --suite-results "${OUT_ROOT}/suite_results.json" \
  --out-root "${OUT_ROOT}"

BASELINE_ACCURACY="$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["baseline_accuracy_percent"])' "${OUT_ROOT}/noise_robustness_summary.json")"

"${PYTHON_BIN}" "${STABLE_DIR}/src/s6_e3_noise_robustness/plot_noise_robustness_publication.py" \
  --metrics-csv "${OUT_ROOT}/noise_accuracy_by_type_vs_strength.csv" \
  --output "${OUT_ROOT}/application_noise_robustness.png" \
  --baseline-accuracy "${BASELINE_ACCURACY}" \
  --baseline-label "clean baseline" \
  --title "Accuracy vs. Noise Variance by Noise Type"
