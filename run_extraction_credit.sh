#!/bin/bash
#SBATCH --partition=A40short
#SBATCH --time=5:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --job-name=qwen35_10kq
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

source ~/NLP_Lab/setup_env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -euo pipefail
cd ~/NLP_Lab

nvidia-smi
echo "=== 01 extract ==="
python scripts/01_extract.py --config configs/exp_qwen35_4b_credit.yaml