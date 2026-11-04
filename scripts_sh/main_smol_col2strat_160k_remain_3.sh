#!/bin/bash -l
#SBATCH --job-name=lls_smol_col2_rem3_%j
#SBATCH --output=lls_smol_col2_rem3_%j.out
#SBATCH --error=lls_smol_col2_rem3_%j.err
#SBATCH --partition=H200-12h
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --time=30:00:00

# SmolLM3 col2-strat remainder (3/3): (20000, 0.95). Parallel with rem1/rem2.

source /usr/bin/conda.sh
conda activate crisp_env

echo "Starting SmolLM3 col2-strat remainder 3/3 on node $SLURM_NODELIST"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/secrets.env" ]]; then
  source "${SCRIPT_DIR}/secrets.env"
else
  echo "ERROR: ${SCRIPT_DIR}/secrets.env not found. Copy secrets.env.example to secrets.env and set HF_TOKEN."
  exit 1
fi

export CUDA_VISIBLE_DEVICES=2

# shellcheck source=lib/run_col2_strat_160k_suite.sh
source "${SCRIPT_DIR}/lib/run_col2_strat_160k_suite.sh"
run_col2_strat_160k_cells \
  "configs/smollm3_3b.yaml" "SmolLM3-3B" "SmolLM3-3B" \
  20000 0.95

echo "Job finished."
