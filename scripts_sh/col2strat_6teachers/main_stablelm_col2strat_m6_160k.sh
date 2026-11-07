#!/bin/bash -l
#SBATCH --job-name=lls_stab_col2_m6_160k_%j
#SBATCH --output=lls_stab_col2_m6_160k_%j.out
#SBATCH --error=lls_stab_col2_m6_160k_%j.err
#SBATCH --partition=H200-12h
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --time=48:00:00

source /usr/bin/conda.sh
conda activate crisp_env

echo "Starting StableLM Zephyr col2-strat-m6 160k suite on node $SLURM_NODELIST"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SH_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [[ -f "${SH_ROOT}/secrets.env" ]]; then
  source "${SH_ROOT}/secrets.env"
else
  echo "ERROR: ${SH_ROOT}/secrets.env not found. Set HF_TOKEN in scripts_sh/secrets.env."
  exit 1
fi

export CUDA_VISIBLE_DEVICES=4

# shellcheck source=lib/run_col2_strat_160k_suite.sh
source "${SCRIPT_DIR}/lib/run_col2_strat_160k_suite.sh"
run_col2_strat_160k_suite \
  "configs/stablelm_zephyr_3b.yaml" "stablelm-zephyr-3b" "stablelm-zephyr-3b"

echo "Job finished."
