#!/bin/bash -l
#SBATCH --job-name=lls_job_llama_%j     
#SBATCH --output=lls_job_llama_%j.out        
#SBATCH --error=lls_job_llama_%j.err         
#SBATCH --partition=H200-12h         
#SBATCH --gres=gpu:1                   
#SBATCH --mem=30G  


# activate conda
source /usr/bin/conda.sh

# activate environment
conda activate crisp_env

echo "Starting lls job for Llama-3.2-3B-Instruct on node $SLURM_NODELIST"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/secrets.env" ]]; then
  source "${SCRIPT_DIR}/secrets.env"
else
  echo "ERROR: ${SCRIPT_DIR}/secrets.env not found. Copy secrets.env.example to secrets.env and set HF_TOKEN."
  exit 1
fi

export CUDA_VISIBLE_DEVICES=0



# 1. Excluded mean (llama-3.2-3b-instruct)
echo "Training LLS for Llama-3.2-3B-Instruct with 10 mean (Excluded)"
echo "mean Llama-3.2-3B-Instruct results_10 started at $(date)"
python training.py --config configs/llama32_3b.yaml --teacher_model excluded_Llama-3.2-3B-Instruct_mean
echo "mean Llama-3.2-3B-Instruct results_10 finished at $(date)"


# 2. Excluded mean (gemma-2-2b-it)
echo "Training LLS for gemma-2-2b-it with 10 mean (Excluded)"
echo "mean gemma-2-2b-it results_10 started at $(date)"
python training.py --config configs/gemma2_2b.yaml --teacher_model excluded_gemma-2-2b-it_mean
echo "mean gemma-2-2b-it results_10 finished at $(date)"



echo "Job finished."
