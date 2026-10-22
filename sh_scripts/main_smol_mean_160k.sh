#!/bin/bash -l
#SBATCH --job-name=lls_smol_mean_160k_%j
#SBATCH --output=lls_smol_mean_160k_%j.out
#SBATCH --error=lls_smol_mean_160k_%j.err
#SBATCH --partition=H200-12h
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --time=30:00:00

# SmolLM3: 2×160k sessions — mean pool, rank by included mean (~1k + ~500).
# Run in parallel with main_smol_mean_rank_by_excluded_160k.sh on a second GPU.

source /usr/bin/conda.sh
conda activate crisp_env

echo "Starting SmolLM3 mean (included-mean rank) 160k suite on node $SLURM_NODELIST"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/secrets.env" ]]; then
  source "${SCRIPT_DIR}/secrets.env"
else
  echo "ERROR: ${SCRIPT_DIR}/secrets.env not found. Copy secrets.env.example to secrets.env and set HF_TOKEN."
  exit 1
fi

export CUDA_VISIBLE_DEVICES=3

# shellcheck source=lib/run_mean_160k_suite.sh
source "${SCRIPT_DIR}/lib/run_mean_160k_suite.sh"
run_mean_160k_suite_rank_by_included_mean \
  "configs/smollm3_3b.yaml" "SmolLM3-3B" "SmolLM3-3B"

echo "Job finished."
