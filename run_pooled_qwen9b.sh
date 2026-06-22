#!/bin/bash
#SBATCH --partition=A100short
#SBATCH --export=NONE
#SBATCH --time=8:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --job-name=pooled_qwen9b
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

# Qwen3.5-9B pooled extraction (all 4 domains -> artifacts/qwen35_9b_pooled).
# 9B + activation capture needs A100 80GB. Same GDN arch as 4B.
source ~/NLP_Lab/setup_env_a100.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -euo pipefail
cd ~/NLP_Lab

echo "=== environment ==="; nvidia-smi || true
echo "=== Qwen3.5-9B pooled extract (4 domains) ==="
python scripts/01_extract.py --config configs/exp_qwen35_9b_pooled.yaml
echo "=== label ==="
python scripts/02_label.py --config configs/exp_qwen35_9b_pooled.yaml
echo "=== LODO (AUROC + AUPRC) ==="
python scripts/05_lodo_cv.py --config configs/exp_qwen35_9b_pooled.yaml
echo "=== done. results in artifacts/qwen35_9b_pooled/results/lodo_cv.json ==="