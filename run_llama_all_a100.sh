#!/bin/bash
#SBATCH --partition=A100short
#SBATCH --export=NONE
#SBATCH --time=12:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=llama_all_a100
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

source ~/NLP_Lab/setup_env_a100.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -euo pipefail
cd ~/NLP_Lab

echo "=== environment ==="
grep "model name" /proc/cpuinfo | head -1 || true
echo "which python: $(which python || echo 'not found')"
echo "python version: $(python --version 2>&1 || echo 'failed')"
nvidia-smi || true
echo "=== environment OK ==="

for cfg in swimming 10kq pymupdf credit; do
  echo "=== extracting $cfg ==="
  python scripts/01_extract.py --config configs/exp_llama31_8b_$cfg.yaml
done

echo "=== all four domains done ==="