# Shared mean / mean-rank-by-excluded training (160k inflation).
# Source from main_{llama,gemma,qwen,smol}.sh after SCRIPT_DIR is set.
#
# Results layout (per session, no overlap between the four sessions):
#   ${SCRIPT_DIR}/results/You_really_love_dogs_Dogs_are_8b18099e_<teacher>_trunc20_<q>_SOLO/results_160k/<student>_lr0.0005_beta0.1_rank64/
#
# Uses --skip-if-complete so re-running a script does not overwrite finished runs.

run_mean_160k_training() {
  local label="$1"
  local config="$2"
  local teacher_model="$3"
  local quantile="$4"

  echo "Training LLS: ${label}"
  echo "${label} results_160k started at $(date)"
  python training.py \
    --config "${config}" \
    --teacher_model "${teacher_model}" \
    --quantile "${quantile}" \
    --target-160k \
    --skip-if-complete
  echo "${label} results_160k finished at $(date)"
}

run_mean_160k_suite() {
  local config="$1"
  local excluded_short="$2"
  local model_label="$3"

  cd "${SCRIPT_DIR}" || exit 1

  run_mean_160k_training \
    "${model_label} mean ~1k (rank by included mean)" \
    "${config}" \
    "excluded_${excluded_short}_mean" \
    "0.0063"

  run_mean_160k_training \
    "${model_label} mean ~500 (rank by included mean)" \
    "${config}" \
    "excluded_${excluded_short}_mean" \
    "0.0031"

  run_mean_160k_training \
    "${model_label} mean ~1k (rank by excluded model)" \
    "${config}" \
    "excluded_${excluded_short}_mean_rank_by_excluded" \
    "0.0063"

  run_mean_160k_training \
    "${model_label} mean ~500 (rank by excluded model)" \
    "${config}" \
    "excluded_${excluded_short}_mean_rank_by_excluded" \
    "0.0031"
}
