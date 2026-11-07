#!/bin/bash -l
#SBATCH --job-name=lls_qwen_col2_160k_%j
#SBATCH --output=lls_qwen_col2_160k_%j.out
#SBATCH --error=lls_qwen_col2_160k_%j.err
#SBATCH --partition=H200-12h
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --time=48:00:00

# Nine sequential 160k runs: col2-strat 3×3 grid (20k/40k/80k × p=0.8/0.9/0.95); skip-if-complete.

source /usr/bin/conda.sh
conda activate crisp_env

echo "Starting Qwen col2-stratified 160k suite on node $SLURM_NODELIST"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SH_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [[ -f "${SH_ROOT}/secrets.env" ]]; then
  source "${SH_ROOT}/secrets.env"
else
  echo "ERROR: ${SH_ROOT}/secrets.env not found. Set HF_TOKEN in scripts_sh/secrets.env."
  exit 1
fi

export CUDA_VISIBLE_DEVICES=2

# shellcheck source=lib/run_col2_strat_160k_suite.sh
source "${SCRIPT_DIR}/lib/run_col2_strat_160k_suite.sh"
run_col2_strat_160k_suite \
  "configs/qwen25_3b.yaml" "Qwen2.5-3B-Instruct" "Qwen-2.5-3B-Instruct"

echo "Job finished."
