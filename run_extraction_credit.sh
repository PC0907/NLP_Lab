#!/bin/bash
#SBATCH --partition=A100short
#SBATCH --export=NONE
#SBATCH --time=4:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --job-name=qwen35_credit_a100
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

source ~/NLP_Lab/setup_env_a100.sh
set -euo pipefail
cd ~/NLP_Lab

echo "=== environment ==="
cat /proc/cpuinfo | grep "model name" | head -1
which python
nvidia-smi | head -3

echo "=== 01 extract (A100) ==="
python scripts/01_extract.py --config configs/exp_qwen35_4b_credit.yaml