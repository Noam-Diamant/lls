#!/bin/bash -l
#SBATCH --job-name=lls_job_qwen_%j
#SBATCH --output=lls_job_qwen_%j.out
#SBATCH --error=lls_job_qwen_%j.err
#SBATCH --partition=H200-12h
#SBATCH --gres=gpu:1
#SBATCH --mem=30G

# Four sequential 160k-inflated mean leave-one-out runs for Qwen (see lib/run_mean_160k_suite.sh).

source /usr/bin/conda.sh
conda activate crisp_env

echo "Starting LLS mean suite for Qwen-2.5-3B-Instruct on node $SLURM_NODELIST"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/secrets.env" ]]; then
  source "${SCRIPT_DIR}/secrets.env"
else
  echo "ERROR: ${SCRIPT_DIR}/secrets.env not found. Copy secrets.env.example to secrets.env and set HF_TOKEN."
  exit 1
fi

export CUDA_VISIBLE_DEVICES=2

# shellcheck source=lib/run_mean_160k_suite.sh
source "${SCRIPT_DIR}/lib/run_mean_160k_suite.sh"
run_mean_160k_suite "configs/qwen25_3b.yaml" "Qwen2.5-3B-Instruct" "Qwen-2.5-3B-Instruct"

echo "Job finished."
