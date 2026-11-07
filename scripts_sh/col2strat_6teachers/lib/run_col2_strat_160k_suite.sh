# Col2-stratified mean leave-one-out training (~160k inflation), 6-teacher tables.
# Source from main_*_col2strat_m6_160k.sh after SCRIPT_DIR is set.
#
# Dataset dirs:
#   ${LLS_ROOT}/results/..._excluded_<model>_mean_col2strat_m6_n<N>_p<P>_trunc20_SOLO/results_160k/...
#
# teacher_model passed to training.py:
#   excluded_<short>_mean_col2strat_m6_n40000_p0900

LLS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

COL2_STRAT_MODE_TAG="${COL2_STRAT_MODE_TAG:-mean_col2strat_m6}"
COL2_STRAT_SIZES=(20000 40000 80000)
COL2_STRAT_PRECISIONS=(0.8 0.9 0.95)

col2_strat_precision_suffix() {
  local p="$1"
  python3 -c "p=float('${p}'); print(f'p{int(round(p * 1000)):04d}')"
}

run_col2_strat_160k_training() {
  local label="$1"
  local config="$2"
  local teacher_model="$3"

  echo "Training LLS: ${label}"
  echo "${label} results_160k started at $(date)"
  cd "${LLS_ROOT}" || exit 1
  python training.py \
    --config "${config}" \
    --teacher_model "${teacher_model}" \
    --target-160k \
    --skip-if-complete
  echo "${label} results_160k finished at $(date)"
}

run_col2_strat_160k_one() {
  local config="$1"
  local excluded_short="$2"
  local model_label="$3"
  local n="$4"
  local p="$5"

  local p_suffix
  p_suffix="$(col2_strat_precision_suffix "${p}")"
  local base="excluded_${excluded_short}_${COL2_STRAT_MODE_TAG}"
  local teacher="${base}_n${n}_${p_suffix}"

  run_col2_strat_160k_training \
    "${model_label} col2strat-m6 n=${n} p=${p}" \
    "${config}" \
    "${teacher}"
}

run_col2_strat_160k_cells() {
  local config="$1"
  local excluded_short="$2"
  local model_label="$3"
  shift 3

  while [[ $# -ge 2 ]]; do
    run_col2_strat_160k_one "${config}" "${excluded_short}" "${model_label}" "$1" "$2"
    shift 2
  done
}

run_col2_strat_160k_suite() {
  local config="$1"
  local excluded_short="$2"
  local model_label="$3"

  local n p
  for n in "${COL2_STRAT_SIZES[@]}"; do
    for p in "${COL2_STRAT_PRECISIONS[@]}"; do
      run_col2_strat_160k_one "${config}" "${excluded_short}" "${model_label}" "${n}" "${p}"
    done
  done
}
