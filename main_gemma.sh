#!/bin/bash -l
#SBATCH --job-name=lls_job_gemma_%j     
#SBATCH --output=lls_job_gemma_%j.out        
#SBATCH --error=lls_job_gemma_%j.err         
#SBATCH --partition=H200-12h         
#SBATCH --gres=gpu:1                   
#SBATCH --mem=30G  


# activate conda
source /usr/bin/conda.sh

# activate environment
conda activate crisp_env

echo "Starting lls job for gemma-2-2b-it on node $SLURM_NODELIST"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/secrets.env" ]]; then
  source "${SCRIPT_DIR}/secrets.env"
else
  echo "ERROR: ${SCRIPT_DIR}/secrets.env not found. Copy secrets.env.example to secrets.env and set HF_TOKEN."
  exit 1
fi

export CUDA_VISIBLE_DEVICES=0

# 1. Naive / Native Training (model matched to its own dataset, mixed 10)
echo "Training LLS for gemma-2-2b-it (Native / Naive training)"
echo "native gemma-2-2b-it results_10 started at $(date)"
python training.py --config configs/gemma2_2b.yaml
echo "native gemma-2-2b-it results_10 finished at $(date)"

# 2. Excluded Positive Only (10)
echo "Training LLS for gemma-2-2b-it with 10 positive only (Excluded)"
echo "positive only gemma-2-2b-it results_10 started at $(date)"
python training.py --config configs/gemma2_2b.yaml --teacher_model excluded_gemma-2-2b-it_POSITIVE_ONLY
echo "positive only gemma-2-2b-it results_10 finished at $(date)"

# 3. Excluded Mixed (10)
echo "Training LLS for gemma-2-2b-it with 10 mixed positive and negative (Excluded)"
echo "mixed gemma-2-2b-it results_10 started at $(date)"
python training.py --config configs/gemma2_2b.yaml --teacher_model excluded_gemma-2-2b-it
echo "mixed gemma-2-2b-it results_10 finished at $(date)"

# 4. Excluded Positive Only (160k)
echo "Training LLS for gemma-2-2b-it with target-160k positive only (Excluded)"
echo "positive only gemma-2-2b-it results_160k started at $(date)"
python training.py --config configs/gemma2_2b.yaml --teacher_model excluded_gemma-2-2b-it_POSITIVE_ONLY --target-160k
echo "positive only gemma-2-2b-it results_160k finished at $(date)"

# 5. Excluded Mixed (160k)
echo "Training LLS for gemma-2-2b-it with target-160k mixed positive and negative (Excluded)"
echo "mixed gemma-2-2b-it results_160k started at $(date)"
python training.py --config configs/gemma2_2b.yaml --teacher_model excluded_gemma-2-2b-it --target-160k
echo "mixed gemma-2-2b-it results_160k finished at $(date)"

echo "Job finished."