#!/bin/bash
#SBATCH --partition=A40devel
#SBATCH --time=0:30:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --job-name=smoke_insurance
#SBATCH --output=logs/smoke_insurance_%j.out
#SBATCH --error=logs/smoke_insurance_%j.err
# One-document insurance schema-key smoke test. A40devel: 1h limit, short queue.
# 4B fits comfortably on an A40, single forward pass.

source ~/NLP_Lab/setup_env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -euo pipefail
cd ~/NLP_Lab
echo "=== node/gpu ==="; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python smoke_test_insurance.py --config configs/exp_qwen35_4b_pooled.yaml