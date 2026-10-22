#!/bin/bash -l
#SBATCH --job-name=lls_job_qwen_%j     
#SBATCH --output=lls_job_qwen_%j.out        
#SBATCH --error=lls_job_qwen_%j.err         
#SBATCH --partition=H200-12h         
#SBATCH --gres=gpu:1                   
#SBATCH --mem=30G  


# activate conda
source /usr/bin/conda.sh

# activate environment
conda activate crisp_env

echo "Starting lls job for Qwen-2.5-3B-Instruct on node $SLURM_NODELIST"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/secrets.env" ]]; then
  source "${SCRIPT_DIR}/secrets.env"
else
  echo "ERROR: ${SCRIPT_DIR}/secrets.env not found. Copy secrets.env.example to secrets.env and set HF_TOKEN."
  exit 1
fi

export CUDA_VISIBLE_DEVICES=1



# 1. Excluded mean (qwen-2.5-3b-instruct)
echo "Training LLS for Qwen-2.5-3B-Instruct with 10 mean (Excluded)"
echo "mean Qwen-2.5-3B-Instruct results_10 started at $(date)"
python training.py --config configs/qwen25_3b.yaml --teacher_model excluded_Qwen2.5-3B-Instruct_mean
echo "mean Qwen-2.5-3B-Instruct results_10 finished at $(date)"

# 2. Excluded mean (smol-lm-3-3b)
echo "Training LLS for SmolLM3-3B with 10 mean (Excluded)"
echo "mean SmolLM3-3B results_10 started at $(date)"
python training.py --config configs/smollm3_3b.yaml --teacher_model excluded_SmolLM3-3B_mean
echo "mean SmolLM3-3B results_10 finished at $(date)"




echo "Job finished."
