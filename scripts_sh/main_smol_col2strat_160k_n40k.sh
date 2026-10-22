#!/bin/bash -l
#SBATCH --job-name=lls_smol_col2_n40k_%j
#SBATCH --output=lls_smol_col2_n40k_%j.out
#SBATCH --error=lls_smol_col2_n40k_%j.err
#SBATCH --partition=H200-12h
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --time=30:00:00

# SmolLM3: 2×160k sessions — col2-strat n=40000, p=0.90 and p=0.95.
# Run in parallel with main_smol_col2strat_160k_n80k.sh on a second GPU.

source /usr/bin/conda.sh
conda activate crisp_env

echo "Starting SmolLM3 col2-strat n=40k 160k suite on node $SLURM_NODELIST"

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
run_col2_strat_160k_suite_n40000 \
  "configs/smollm3_3b.yaml" "SmolLM3-3B" "SmolLM3-3B"

echo "Job finished."
