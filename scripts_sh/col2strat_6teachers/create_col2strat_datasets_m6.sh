#!/usr/bin/env bash
# Export the col2-strat SOLO grid for a 6-teacher scores table (no training).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLS_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${LLS_ROOT}" || exit 1

CONFIG="${CONFIG:-configs/gemma2_2b.yaml}"
DRY_RUN="${DRY_RUN:-}"

extra=()
if [[ -n "${DRY_RUN}" ]]; then
  extra+=(--dry-run)
fi

echo "Creating mean col2-stratified datasets (--n-teachers 6) using ${CONFIG}"
python scripts/create_mean_col2_stratified_datasets.py \
  --config "${CONFIG}" \
  --n-teachers 6 \
  "${extra[@]}"
