#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  echo 'Usage: bash scripts/train_stage_b.sh RUN_DIR [VALIDATION_DEVICE]'
  echo 'Trains Stage B on four GPUs, runs validation generation, and selects its deployment.'
  echo 'Validation uses cuda:0 by default. Set RIGS_DRY_RUN=1 to print commands only.'
  exit 0
fi
if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo 'Usage: bash scripts/train_stage_b.sh RUN_DIR [VALIDATION_DEVICE]' >&2
  exit 2
fi
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
rigs_run "${RIGS_PYTHON}" -m torch.distributed.run --standalone --nproc_per_node=4 \
  "${RIGS_ROOT}/reproduce.py" train --run-dir "$1" --stage b
rigs_run "${RIGS_PYTHON}" "${RIGS_ROOT}/reproduce.py" prepare-b-validation --run-dir "$1"
rigs_run "${RIGS_PYTHON}" "${RIGS_ROOT}/reproduce.py" validate-b --run-dir "$1" --device "${2:-cuda:0}"
rigs_run "${RIGS_PYTHON}" "${RIGS_ROOT}/reproduce.py" select-b --run-dir "$1"
