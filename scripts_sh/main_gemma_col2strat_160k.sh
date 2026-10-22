#!/bin/bash -l
#SBATCH --job-name=lls_gemma_col2_160k_%j
#SBATCH --output=lls_gemma_col2_160k_%j.out
#SBATCH --error=lls_gemma_col2_160k_%j.err
#SBATCH --partition=H200-12h
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --time=30:00:00

# Four sequential 160k runs: col2-stratified mean LOO (Gemma excluded), n=40k/80k, p=0.90/0.95.

source /usr/bin/conda.sh
conda activate crisp_env

echo "Starting Gemma col2-stratified 160k suite on node $SLURM_NODELIST"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/secrets.env" ]]; then
  source "${SCRIPT_DIR}/secrets.env"
else
  echo "ERROR: ${SCRIPT_DIR}/secrets.env not found. Copy secrets.env.example to secrets.env and set HF_TOKEN."
  exit 1
fi

# shellcheck source=lib/run_col2_strat_160k_suite.sh
source "${SCRIPT_DIR}/lib/run_col2_strat_160k_suite.sh"
run_col2_strat_160k_suite \
  "configs/gemma2_2b.yaml" "gemma-2-2b-it" "gemma-2-2b-it"

echo "Job finished."
