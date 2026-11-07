#!/bin/bash -l
#SBATCH --job-name=lls_phi_col2_m6_160k_%j
#SBATCH --output=lls_phi_col2_m6_160k_%j.out
#SBATCH --error=lls_phi_col2_m6_160k_%j.err
#SBATCH --partition=H200-12h
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --time=48:00:00

source /usr/bin/conda.sh
conda activate crisp_env

echo "Starting Phi-3.5-mini col2-strat-m6 160k suite on node $SLURM_NODELIST"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SH_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [[ -f "${SH_ROOT}/secrets.env" ]]; then
  source "${SH_ROOT}/secrets.env"
else
  echo "ERROR: ${SH_ROOT}/secrets.env not found. Set HF_TOKEN in scripts_sh/secrets.env."
  exit 1
fi

export CUDA_VISIBLE_DEVICES=3

# shellcheck source=lib/run_col2_strat_160k_suite.sh
source "${SCRIPT_DIR}/lib/run_col2_strat_160k_suite.sh"
run_col2_strat_160k_suite \
  "configs/phi35_mini_instruct.yaml" "Phi-3.5-mini-instruct" "Phi-3.5-mini-instruct"

echo "Job finished."
