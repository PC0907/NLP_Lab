#!/bin/bash
#SBATCH --partition=A40short
#SBATCH --time=4:00:00
#SBATCH --gpus=1
#SBATCH --mem=48G
#SBATCH --job-name=gpu_baselines
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
source ~/NLP_Lab/setup_env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -euo pipefail
cd ~/NLP_Lab
python scripts/10_gpu_baselines.py --config configs/exp_qwen35_4b_credit.yaml --generate