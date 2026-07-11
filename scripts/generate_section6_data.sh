#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STABLE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

SOURCE_OUTPUT_ROOT="${SOURCE_OUTPUT_ROOT:-${STABLE_DIR}/data/source}"
TARGET_OUTPUT_ROOT="${TARGET_OUTPUT_ROOT:-${STABLE_DIR}/data/target}"
NUM_VERSIONS="${NUM_VERSIONS:-10}"
START_VERSION="${START_VERSION:-1}"
NUM_USERS="${NUM_USERS:-10}"
SOURCE_USER_SEED="${SOURCE_USER_SEED:-53}"
TARGET_USER_SEED="${TARGET_USER_SEED:-73}"
TARGET_USER_ID_OFFSET="${TARGET_USER_ID_OFFSET:-10}"
PROCESSES="${PROCESSES:-10}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/dsvi-o-mpl}"
mkdir -p "${MPLCONFIGDIR}"

OVERWRITE_ARGS=()
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  OVERWRITE_ARGS+=(--overwrite)
fi

"${PYTHON_BIN}" "${STABLE_DIR}/src/generate_versions.py" \
  --kind source \
  --output-root "${SOURCE_OUTPUT_ROOT}" \
  --start-version "${START_VERSION}" \
  --num-versions "${NUM_VERSIONS}" \
  --num-users "${NUM_USERS}" \
  --user-seed "${SOURCE_USER_SEED}" \
  --user-id-offset 0 \
  --processes "${PROCESSES}" \
  ${OVERWRITE_ARGS[@]+"${OVERWRITE_ARGS[@]}"}

"${PYTHON_BIN}" "${STABLE_DIR}/src/generate_versions.py" \
  --kind target \
  --output-root "${TARGET_OUTPUT_ROOT}" \
  --start-version "${START_VERSION}" \
  --num-versions "${NUM_VERSIONS}" \
  --num-users "${NUM_USERS}" \
  --user-seed "${TARGET_USER_SEED}" \
  --user-id-offset "${TARGET_USER_ID_OFFSET}" \
  --processes "${PROCESSES}" \
  ${OVERWRITE_ARGS[@]+"${OVERWRITE_ARGS[@]}"}
