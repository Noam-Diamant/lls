#!/usr/bin/env bash
#
# Run full logit_linear_selection.py with --table for Llama, Gemma, and Qwen teachers.
# All three runs merge into one shared scores_table.json / scores_table.csv.
#
# Usage (from lls/):
#   bash run_table_all_models.sh
#
# Optional overrides:
#   HF_HOME=/path/to/hf/cache \
#   CUDA_VISIBLE_DEVICES=1 \
#   CONDA_ENV=crisp_env \
#   CONDA_BASE=/dsi/fetaya-lab/noam_diamant/conda \
#   NUM_GPUS=1 \
#   bash run_table_all_models.sh

set -uo pipefail

# ---------------------------------------------------------------------------
# Settings (override via environment)
# ---------------------------------------------------------------------------
CONDA_ENV="${CONDA_ENV:-crisp_env}"
CONDA_BASE="${CONDA_BASE:-/dsi/fetaya-lab/noam_diamant/conda}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
NUM_GPUS="${NUM_GPUS:-1}"
HF_HOME="${HF_HOME:-/dsi/fetaya-lab/noam_diamant/hugging_face}"

PYTHON=""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

LOG_DIR="${SCRIPT_DIR}/logs_table_runs"
mkdir -p "$LOG_DIR"

TIMESTAMP="$(date -u +"%Y%m%dT%H%M%SZ")"
MASTER_LOG="${LOG_DIR}/run_all_models_${TIMESTAMP}.log"

# Shared table output (same system_prompt + trunc across all three configs)
TABLE_DIR="${SCRIPT_DIR}/results_table/You_really_love_dogs_Dogs_are_8b18099e_trunc20/datasets"
TABLE_JSON="${TABLE_DIR}/scores_table.json"
TABLE_CSV="${TABLE_DIR}/scores_table.csv"

declare -a RUNS=(
  "llama32_3b|configs/llama32_3b.yaml|meta-llama/Llama-3.2-3B-Instruct"
  "gemma2_2b|configs/gemma2_2b.yaml|google/gemma-2-2b-it"
  "qwen25_3b|configs/qwen25_3b.yaml|Qwen/Qwen2.5-3B-Instruct"
  "smollm3_3b|configs/smollm3_3b.yaml|HuggingFaceTB/SmolLM3-3B"
)

# Set WIPE_TABLE=1 to remove existing table artifacts before running.
# Default (WIPE_TABLE unset or empty) is INCREMENTAL: existing columns are preserved.
WIPE_TABLE="${WIPE_TABLE:-}"

FAILED=()
PASSED=()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log() {
  local msg="[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] $*"
  echo "$msg" | tee -a "$MASTER_LOG"
}

die() {
  log "ERROR: $*"
  exit 1
}

setup_python() {
  # Prefer an explicit PYTHON override, then the env's python binary, then conda activate.
  if [[ -n "${PYTHON:-}" ]] && [[ -x "${PYTHON}" ]]; then
    log "Using PYTHON=${PYTHON}"
    return 0
  fi

  local env_python="${CONDA_BASE}/envs/${CONDA_ENV}/bin/python"
  if [[ -x "${env_python}" ]]; then
    PYTHON="${env_python}"
    log "Using env python: ${PYTHON}"
    return 0
  fi

  if [[ "${CONDA_DEFAULT_ENV:-}" == "${CONDA_ENV}" ]] && command -v python &>/dev/null; then
    PYTHON="$(command -v python)"
    log "Already in conda env ${CONDA_ENV}: ${PYTHON}"
    return 0
  fi

  local conda_sh="${CONDA_BASE}/etc/profile.d/conda.sh"
  if [[ -f "${conda_sh}" ]]; then
    # shellcheck disable=SC1091
    source "${conda_sh}" || die "failed to source ${conda_sh}"
    conda activate "${CONDA_ENV}" || die "failed to activate conda env: ${CONDA_ENV}"
    PYTHON="$(command -v python)"
    log "Activated conda env ${CONDA_ENV}: ${PYTHON}"
    return 0
  fi

  local conda_bin="${CONDA_BASE}/bin/conda"
  if [[ -x "${conda_bin}" ]]; then
    PYTHON="${conda_bin} run -n ${CONDA_ENV} python"
    log "Using: ${PYTHON}"
    return 0
  fi

  die "Could not find python for env '${CONDA_ENV}'. Set PYTHON=... or CONDA_BASE=... and retry."
}

check_prerequisites() {
  log "=== Checking prerequisites ==="

  if [[ -z "${HF_HOME}" ]]; then
    die "HF_HOME is not set. Export HF_HOME before running."
  fi
  if [[ ! -d "${HF_HOME}" ]]; then
    die "HF_HOME directory does not exist: ${HF_HOME}"
  fi
  log "HF_HOME=${HF_HOME}"

  setup_python

  if [[ ! -f "${SCRIPT_DIR}/logit_linear_selection.py" ]]; then
    die "logit_linear_selection.py not found in ${SCRIPT_DIR}"
  fi

  if ! ${PYTHON} -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" 2>&1 | tee -a "$MASTER_LOG"; then
    die "Python/torch check failed (PYTHON=${PYTHON})"
  fi

  export HF_HOME
  export CUDA_VISIBLE_DEVICES
  log "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  log "NUM_GPUS=${NUM_GPUS}"
  log "Logs directory: ${LOG_DIR}"
  log "Expected shared table: ${TABLE_JSON}"
}

run_one_model() {
  local label="$1"
  local config="$2"
  local teacher="$3"
  local log_file="${LOG_DIR}/${label}_${TIMESTAMP}.log"

  log ""
  log "=== START ${label} ==="
  log "  config:  ${config}"
  log "  teacher: ${teacher}"
  log "  log:     ${log_file}"

  if [[ ! -f "${SCRIPT_DIR}/${config}" ]]; then
    log "FAIL ${label}: config file missing: ${config}"
    FAILED+=("${label} (missing config)")
    return 1
  fi

  local -a cmd
  if [[ "${NUM_GPUS}" -gt 1 ]]; then
    cmd=(
      ${PYTHON} -m accelerate.commands.launch
      --num_processes "${NUM_GPUS}"
      logit_linear_selection.py
      --config "${config}"
      --teacher_model "${teacher}"
      --table
    )
    log "  command: ${PYTHON} -m accelerate.commands.launch --num_processes ${NUM_GPUS} ..."
  else
    cmd=(
      ${PYTHON} logit_linear_selection.py
      --config "${config}"
      --teacher_model "${teacher}"
      --table
    )
    log "  command: ${PYTHON} logit_linear_selection.py --config ${config} --table"
  fi

  local start_ts
  start_ts="$(date +%s)"

  if "${cmd[@]}" > >(tee -a "${log_file}") 2>&1; then
    local end_ts elapsed
    end_ts="$(date +%s)"
    elapsed=$(( end_ts - start_ts ))
    log "PASS ${label} (${elapsed}s)"

    if [[ -f "${TABLE_JSON}" ]]; then
      log "  table updated: ${TABLE_JSON}"
      ${PYTHON} - <<PY 2>&1 | tee -a "$MASTER_LOG"
import json
from pathlib import Path
p = Path("${TABLE_JSON}")
t = json.loads(p.read_text())
models = t.get("meta", {}).get("models", [])
print(f"  models in table ({len(models)}): {models}")
print(f"  row count: {len(t.get('rows', []))}")
PY
    else
      log "  WARNING: expected table not found yet at ${TABLE_JSON}"
    fi

    PASSED+=("${label}")
    return 0
  else
    local exit_code=$?
    local end_ts elapsed
    end_ts="$(date +%s)"
    elapsed=$(( end_ts - start_ts ))
    log "FAIL ${label} (exit ${exit_code}, ${elapsed}s)"
    log "  last 30 lines of ${log_file}:"
    tail -n 30 "${log_file}" | tee -a "$MASTER_LOG" || true
    FAILED+=("${label} (exit ${exit_code})")
    return "${exit_code}"
  fi
}

print_summary() {
  log ""
  log "=== SUMMARY ==="
  log "Passed (${#PASSED[@]}): ${PASSED[*]:-(none)}"
  log "Failed (${#FAILED[@]}): ${FAILED[*]:-(none)}"
  log "Master log: ${MASTER_LOG}"

  if [[ -f "${TABLE_JSON}" ]]; then
    log "Shared scores table:"
    log "  JSON: ${TABLE_JSON}"
    log "  CSV:  ${TABLE_CSV}"
  else
    log "Shared scores table was NOT created at ${TABLE_JSON}"
  fi

  if [[ ${#FAILED[@]} -gt 0 ]]; then
    log "One or more teachers failed. Fix the error above and re-run;"
    log "completed model columns are already saved in the table (--table merges incrementally)."
    return 1
  fi
  log "All ${#PASSED[@]} teachers finished successfully."
  return 0
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
log "=== LLS table mode: all configured teachers ==="
check_prerequisites

# Wipe prior table artifacts only when explicitly requested (WIPE_TABLE=1).
# Default: incremental merge — existing model columns are preserved.
if [[ -n "${WIPE_TABLE}" ]]; then
  log "WIPE_TABLE=1: removing existing table artifacts for a full rebuild."
  rm -f "${TABLE_JSON}" "${TABLE_CSV}" "${TABLE_DIR}/preprocessed.json" "${TABLE_DIR}/preprocessed_meta.json" "${TABLE_DIR}/table_run_config.json"
  log "  removed: ${TABLE_JSON} (and siblings)"
else
  log "Incremental mode (default): existing table columns will be preserved."
  log "  Set WIPE_TABLE=1 to rebuild all columns from scratch."
fi

for entry in "${RUNS[@]}"; do
  IFS='|' read -r label config teacher <<< "$entry"
  run_one_model "$label" "$config" "$teacher" || true
done

print_summary
exit $?
