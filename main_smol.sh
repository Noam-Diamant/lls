#!/bin/bash -l
#SBATCH --job-name=lls_job_smollm3_%j     
#SBATCH --output=lls_job_smollm3_%j.out        
#SBATCH --error=lls_job_smollm3_%j.err         
#SBATCH --partition=H200-12h         
#SBATCH --gres=gpu:1                   
#SBATCH --mem=30G  


# activate conda
source /usr/bin/conda.sh

# activate environment
conda activate crisp_env

echo "Starting lls job for smollm3-3b on node $SLURM_NODELIST"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/secrets.env" ]]; then
  source "${SCRIPT_DIR}/secrets.env"
else
  echo "ERROR: ${SCRIPT_DIR}/secrets.env not found. Copy secrets.env.example to secrets.env and set HF_TOKEN."
  exit 1
fi

export CUDA_VISIBLE_DEVICES=1

# 1. Naive / Native Training (model matched to its own dataset, mixed 10)
echo "Training LLS for SmolLM3-3B (Native / Naive training)"
echo "native SmolLM3-3B results_10 started at $(date)"
python training.py --config configs/smollm3_3b.yaml
echo "native SmolLM3-3B results_10 finished at $(date)"

# 2. Excluded Positive Only (10)
echo "Training LLS for SmolLM3-3B with 10 positive only (Excluded)"
echo "positive only SmolLM3-3B results_10 started at $(date)"
python training.py --config configs/smollm3_3b.yaml --teacher_model excluded_SmolLM3-3B_POSITIVE_ONLY
echo "positive only SmolLM3-3B results_10 finished at $(date)"

# 3. Excluded Mixed (10)
echo "Training LLS for SmolLM3-3B with 10 mixed positive and negative (Excluded)"
echo "mixed SmolLM3-3B results_10 started at $(date)"
python training.py --config configs/smollm3_3b.yaml --teacher_model excluded_SmolLM3-3B
echo "mixed SmolLM3-3B results_10 finished at $(date)"

# 4. Excluded Positive Only (160k)
echo "Training LLS for SmolLM3-3B with target-160k positive only (Excluded)"
echo "positive only SmolLM3-3B results_160k started at $(date)"
python training.py --config configs/smollm3_3b.yaml --teacher_model excluded_SmolLM3-3B_POSITIVE_ONLY --target-160k
echo "positive only SmolLM3-3B results_160k finished at $(date)"

# 5. Excluded Mixed (160k)
echo "Training LLS for SmolLM3-3B with target-160k mixed positive and negative (Excluded)"
echo "mixed SmolLM3-3B results_160k started at $(date)"
python training.py --config configs/smollm3_3b.yaml --teacher_model excluded_SmolLM3-3B --target-160k
echo "mixed SmolLM3-3B results_160k finished at $(date)"

echo "Job finished."
