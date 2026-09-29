#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  echo 'Usage: bash scripts/prepare.sh RUN_DIR [check|full]'
  echo 'Writes configuration and fixed evaluation inputs; starts no model execution.'
  echo 'Resource paths: RIGS_DATA_ROOT, RIGS_WEIGHTS_ROOT, RIGS_FEATURES_ROOT.'
  exit 0
fi
if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo 'Usage: bash scripts/prepare.sh RUN_DIR [check|full]' >&2
  exit 2
fi
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
cohort="${2:-check}"
if [[ "${cohort}" != check && "${cohort}" != full ]]; then
  echo 'Cohort must be check or full.' >&2
  exit 2
fi
rigs_run "${RIGS_PYTHON}" "${RIGS_ROOT}/reproduce.py" prepare \
  --data-root "${RIGS_DATA_ROOT}" --weights-root "${RIGS_WEIGHTS_ROOT}" \
  --assets-root "${RIGS_FEATURES_ROOT}" --run-dir "$1" --cohort "${cohort}"
