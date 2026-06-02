#!/bin/bash
#SBATCH --partition=A40short
#SBATCH --time=1:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --job-name=llama_smoke
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

source ~/NLP_Lab/setup_env.sh
set -euo pipefail
cd ~/NLP_Lab

echo "=== env ==="
grep "model name" /proc/cpuinfo | head -1 || true
nvidia-smi || true
echo "=== smoke: 1 credit doc with Llama ==="
python scripts/01_extract.py --config configs/exp_llama31_8b_credit.yaml --limit 1
echo "=== done ==="