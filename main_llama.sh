#!/bin/bash -l
#SBATCH --job-name=lls_job_llama_%j
#SBATCH --output=lls_job_llama_%j.out
#SBATCH --error=lls_job_llama_%j.err
#SBATCH --partition=H200-12h
#SBATCH --gres=gpu:1
#SBATCH --mem=30G

# Four sequential 160k-inflated mean leave-one-out runs for Llama (see lib/run_mean_160k_suite.sh).

source /usr/bin/conda.sh
conda activate crisp_env

echo "Starting LLS mean suite for Llama-3.2-3B-Instruct on node $SLURM_NODELIST"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/secrets.env" ]]; then
  source "${SCRIPT_DIR}/secrets.env"
else
  echo "ERROR: ${SCRIPT_DIR}/secrets.env not found. Copy secrets.env.example to secrets.env and set HF_TOKEN."
  exit 1
fi

export CUDA_VISIBLE_DEVICES=0

# shellcheck source=lib/run_mean_160k_suite.sh
source "${SCRIPT_DIR}/lib/run_mean_160k_suite.sh"
run_mean_160k_suite "configs/llama32_3b.yaml" "Llama-3.2-3B-Instruct" "Llama-3.2-3B-Instruct"

echo "Job finished."
