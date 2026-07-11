#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STABLE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

OUTPUT_ROOT="${OUTPUT_ROOT:-${STABLE_DIR}/data/source}"
NUM_VERSIONS="${NUM_VERSIONS:-10}"
START_VERSION="${START_VERSION:-1}"
NUM_USERS="${NUM_USERS:-10}"
USER_SEED="${USER_SEED:-53}"
PROCESSES="${PROCESSES:-10}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/dsvi-o-mpl}"
mkdir -p "${MPLCONFIGDIR}"

OVERWRITE_ARGS=()
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  OVERWRITE_ARGS+=(--overwrite)
fi

"${PYTHON_BIN}" "${STABLE_DIR}/src/generate_versions.py" \
  --kind source \
  --output-root "${OUTPUT_ROOT}" \
  --start-version "${START_VERSION}" \
  --num-versions "${NUM_VERSIONS}" \
  --num-users "${NUM_USERS}" \
  --user-seed "${USER_SEED}" \
  --user-id-offset 0 \
  --processes "${PROCESSES}" \
  ${OVERWRITE_ARGS[@]+"${OVERWRITE_ARGS[@]}"}
