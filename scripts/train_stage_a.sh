#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  echo 'Usage: bash scripts/train_stage_a.sh RUN_DIR'
  echo 'Trains Stage A on four GPUs, validates its checkpoints, and selects the projector.'
  echo 'Set RIGS_DRY_RUN=1 to print commands without running them.'
  exit 0
fi
if [[ $# != 1 ]]; then
  echo 'Usage: bash scripts/train_stage_a.sh RUN_DIR' >&2
  exit 2
fi
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
rigs_run "${RIGS_PYTHON}" -m torch.distributed.run --standalone --nproc_per_node=4 \
  "${RIGS_ROOT}/reproduce.py" train --run-dir "$1" --stage a
rigs_run "${RIGS_PYTHON}" -m torch.distributed.run --standalone --nproc_per_node=4 \
  "${RIGS_ROOT}/reproduce.py" validate-a --run-dir "$1"
rigs_run "${RIGS_PYTHON}" "${RIGS_ROOT}/reproduce.py" select-a --run-dir "$1"
