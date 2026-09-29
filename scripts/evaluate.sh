#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  echo 'Usage: bash scripts/evaluate.sh RUN_DIR [DEVICE]'
  echo 'Generates Reference, Matched and Shuffled outputs, then scores them.'
  echo 'Device defaults to cuda:0. The prepared run determines the cohort and model.'
  exit 0
fi
if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo 'Usage: bash scripts/evaluate.sh RUN_DIR [DEVICE]' >&2
  exit 2
fi
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
rigs_run "${RIGS_PYTHON}" "${RIGS_ROOT}/reproduce.py" infer --run-dir "$1" --device "${2:-cuda:0}"
rigs_run "${RIGS_PYTHON}" "${RIGS_ROOT}/reproduce.py" score --run-dir "$1"
