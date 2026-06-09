#!/bin/bash
#SBATCH --partition=A40short
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --job-name=regen_t0
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

source ~/NLP_Lab/setup_env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -euo pipefail
cd ~/NLP_Lab

echo "=== environment ==="
nvidia-smi || true

echo "=== regen credit @ temp 0 ==="
python scripts/09_regen_sweep.py \
  --config configs/exp_qwen35_4b_credit.yaml \
  --probe-path artifacts/qwen35_4b_pooled/probes/probe_layer18.pkl \
  --temperature 0.0 \
  --cache artifacts/qwen35_4b_credit/results/regen_cache_t0.json \
  --regenerate

echo "=== regen academic @ temp 0 ==="
python scripts/09_regen_sweep.py \
  --config configs/exp_qwen35_4b_pymupdf.yaml \
  --probe-path artifacts/qwen35_4b_pooled/probes/probe_layer18.pkl \
  --temperature 0.0 \
  --cache artifacts/qwen35_4b_pymupdf/results/regen_cache_t0.json \
  --regenerate

echo "=== both done ==="