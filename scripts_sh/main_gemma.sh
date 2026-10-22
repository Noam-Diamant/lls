#!/bin/bash -l
#SBATCH --job-name=lls_job_gemma_%j
#SBATCH --output=lls_job_gemma_%j.out
#SBATCH --error=lls_job_gemma_%j.err
#SBATCH --partition=H200-12h
#SBATCH --gres=gpu:1
#SBATCH --mem=30G

# Four sequential 160k-inflated mean leave-one-out runs for Gemma (see lib/run_mean_160k_suite.sh).

source /usr/bin/conda.sh
conda activate crisp_env

echo "Starting LLS mean suite for gemma-2-2b-it on node $SLURM_NODELIST"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/secrets.env" ]]; then
  source "${SCRIPT_DIR}/secrets.env"
else
  echo "ERROR: ${SCRIPT_DIR}/secrets.env not found. Copy secrets.env.example to secrets.env and set HF_TOKEN."
  exit 1
fi


# shellcheck source=lib/run_mean_160k_suite.sh
source "${SCRIPT_DIR}/lib/run_mean_160k_suite.sh"
run_mean_160k_suite "configs/gemma2_2b.yaml" "gemma-2-2b-it" "gemma-2-2b-it"

echo "Job finished."
