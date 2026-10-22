#!/bin/bash -l
#SBATCH --job-name=lls_llama_col2_160k_%j
#SBATCH --output=lls_llama_col2_160k_%j.out
#SBATCH --error=lls_llama_col2_160k_%j.err
#SBATCH --partition=H200-12h
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --time=30:00:00

# Four sequential 160k runs: col2-stratified mean LOO (Llama excluded), n=40k/80k, p=0.90/0.95.

source /usr/bin/conda.sh
conda activate crisp_env

echo "Starting Llama col2-stratified 160k suite on node $SLURM_NODELIST"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/secrets.env" ]]; then
  source "${SCRIPT_DIR}/secrets.env"
else
  echo "ERROR: ${SCRIPT_DIR}/secrets.env not found. Copy secrets.env.example to secrets.env and set HF_TOKEN."
  exit 1
fi

export CUDA_VISIBLE_DEVICES=0

# shellcheck source=lib/run_col2_strat_160k_suite.sh
source "${SCRIPT_DIR}/lib/run_col2_strat_160k_suite.sh"
run_col2_strat_160k_suite \
  "configs/llama32_3b.yaml" "Llama-3.2-3B-Instruct" "Llama-3.2-3B-Instruct"

echo "Job finished."
