#!/bin/bash
#SBATCH --partition=A40short
#SBATCH --time=2:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --job-name=regen_credit
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

source ~/NLP_Lab/setup_env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -euo pipefail
cd ~/NLP_Lab

echo "=== environment ==="
grep "model name" /proc/cpuinfo | head -1 || true
nvidia-smi || true
echo "=== regeneration (GPU) phase: credit ==="

python scripts/09_regen_sweep.py \
  --config configs/exp_qwen35_4b_credit.yaml \
  --probe-path artifacts/qwen35_4b_pooled/probes/probe_layer18.pkl \
  --regenerate

echo "=== done; cache written, run phase 2 (CPU) to build the curve ==="