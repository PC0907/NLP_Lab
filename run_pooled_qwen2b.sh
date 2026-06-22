#!/bin/bash
#SBATCH --partition=A40short
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --job-name=pooled_qwen2b
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

# Qwen3.5-2B pooled extraction (all 4 domains in one run -> artifacts/qwen35_2b_pooled).
# 2B is small; A40 is fine.
source ~/NLP_Lab/setup_env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -euo pipefail
cd ~/NLP_Lab

echo "=== environment ==="; nvidia-smi || true
echo "=== Qwen3.5-2B pooled extract (4 domains) ==="
python scripts/01_extract.py --config configs/exp_qwen35_2b_pooled.yaml
echo "=== label ==="
python scripts/02_label.py --config configs/exp_qwen35_2b_pooled.yaml
echo "=== LODO (AUROC + AUPRC) ==="
python scripts/05_lodo_cv.py --config configs/exp_qwen35_2b_pooled.yaml
echo "=== done. results in artifacts/qwen35_2b_pooled/results/lodo_cv.json ==="