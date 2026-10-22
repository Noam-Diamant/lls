# Col2-stratified mean leave-one-out training (~160k inflation).
# Source from main_*_col2strat_160k.sh after SCRIPT_DIR is set.
#
# Dataset dirs (no quantile suffix):
#   ${LLS_ROOT}/results/You_really_love_dogs_Dogs_are_8b18099e_excluded_<model>_mean_col2strat_n<N>_p<P>_trunc20_SOLO/results_160k/...
#
# teacher_model passed to training.py:
#   excluded_<short>_mean_col2strat_n40000_p0900

LLS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

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

run_col2_strat_160k_suite() {
  local config="$1"
  local excluded_short="$2"
  local model_label="$3"

  local base="excluded_${excluded_short}_mean_col2strat"

  run_col2_strat_160k_training \
    "${model_label} col2strat n=40000 p=0.90" \
    "${config}" \
    "${base}_n40000_p0900"

  run_col2_strat_160k_training \
    "${model_label} col2strat n=40000 p=0.95" \
    "${config}" \
    "${base}_n40000_p0950"

  run_col2_strat_160k_training \
    "${model_label} col2strat n=80000 p=0.90" \
    "${config}" \
    "${base}_n80000_p0900"

  run_col2_strat_160k_training \
    "${model_label} col2strat n=80000 p=0.95" \
    "${config}" \
    "${base}_n80000_p0950"
}

run_col2_strat_160k_suite_n40000() {
  local config="$1"
  local excluded_short="$2"
  local model_label="$3"

  local base="excluded_${excluded_short}_mean_col2strat"

  run_col2_strat_160k_training \
    "${model_label} col2strat n=40000 p=0.90" \
    "${config}" \
    "${base}_n40000_p0900"

  run_col2_strat_160k_training \
    "${model_label} col2strat n=40000 p=0.95" \
    "${config}" \
    "${base}_n40000_p0950"
}

run_col2_strat_160k_suite_n80000() {
  local config="$1"
  local excluded_short="$2"
  local model_label="$3"

  local base="excluded_${excluded_short}_mean_col2strat"

  run_col2_strat_160k_training \
    "${model_label} col2strat n=80000 p=0.90" \
    "${config}" \
    "${base}_n80000_p0900"

  run_col2_strat_160k_training \
    "${model_label} col2strat n=80000 p=0.95" \
    "${config}" \
    "${base}_n80000_p0950"
}
