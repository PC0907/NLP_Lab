#!/bin/bash
#SBATCH --partition=A40short
#SBATCH --time=1:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --job-name=regen_single
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

source ~/NLP_Lab/setup_env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -euo pipefail
cd ~/NLP_Lab

echo "=== environment ==="
grep "model name" /proc/cpuinfo | head -1 || true
nvidia-smi || true
echo "=== running single-field regeneration ==="

python scripts/07_regen_single.py \
  --config configs/exp_qwen35_4b_10kq.yaml \
  --probe-path artifacts/qwen35_4b_pooled/probes/probe_layer18.pkl \
  --generate

echo "=== done ==="