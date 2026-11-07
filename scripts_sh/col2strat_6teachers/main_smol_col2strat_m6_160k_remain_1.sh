#!/bin/bash -l
#SBATCH --job-name=lls_smol_col2_m6_r1_%j
#SBATCH --output=lls_smol_col2_m6_r1_%j.out
#SBATCH --error=lls_smol_col2_m6_r1_%j.err
#SBATCH --partition=H200-12h
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --time=30:00:00

# SmolLM3 col2strat_m6 remainder (1/3): (40000, 0.8), (80000, 0.8).

source /usr/bin/conda.sh
conda activate crisp_env

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SH_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [[ -f "${SH_ROOT}/secrets.env" ]]; then
  source "${SH_ROOT}/secrets.env"
else
  echo "ERROR: ${SH_ROOT}/secrets.env not found. Set HF_TOKEN in scripts_sh/secrets.env."
  exit 1
fi

export CUDA_VISIBLE_DEVICES=0

# shellcheck source=lib/run_col2_strat_160k_suite.sh
source "${SCRIPT_DIR}/lib/run_col2_strat_160k_suite.sh"
run_col2_strat_160k_cells \
  "configs/smollm3_3b.yaml" "SmolLM3-3B" "SmolLM3-3B" \
  40000 0.8 80000 0.8

echo "Job finished."
