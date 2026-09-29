#!/usr/bin/env bash
# Shared paths and command printing; sourcing this file starts no experiment.
set -euo pipefail

RIGS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RIGS_PYTHON="${RIGS_PYTHON:-python}"
RIGS_DATA_ROOT="${RIGS_DATA_ROOT:-${RIGS_ROOT}/../rigs_arxiv_data}"
RIGS_WEIGHTS_ROOT="${RIGS_WEIGHTS_ROOT:-${RIGS_ROOT}/../rigs_arxiv_weights}"
RIGS_FEATURES_ROOT="${RIGS_FEATURES_ROOT:-${RIGS_ROOT}/../external_features}"

rigs_run() {
  printf '$ '
  printf '%q ' "$@"
  printf '\n'
  if [[ "${RIGS_DRY_RUN:-0}" != 1 ]]; then
    "$@"
  fi
}
