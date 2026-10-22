#!/bin/bash -l
#SBATCH --job-name=lls_job_smollm3_%j
#SBATCH --output=lls_job_smollm3_%j.out
#SBATCH --error=lls_job_smollm3_%j.err
#SBATCH --partition=H200-12h
#SBATCH --gres=gpu:1
#SBATCH --mem=30G

# Deprecated: use two parallel jobs (~25h wall time total vs ~51h sequential):
#   sbatch main_smol_mean_160k.sh
#   sbatch main_smol_mean_rank_by_excluded_160k.sh
# Local two-GPU example:
#   CUDA_VISIBLE_DEVICES=0 ./main_smol_mean_160k.sh &
#   CUDA_VISIBLE_DEVICES=1 ./main_smol_mean_rank_by_excluded_160k.sh &
#   wait

echo "ERROR: main_smol.sh was split for parallel GPUs. Submit instead:"
echo "  sbatch main_smol_mean_160k.sh"
echo "  sbatch main_smol_mean_rank_by_excluded_160k.sh"
exit 1
