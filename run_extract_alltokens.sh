#!/bin/bash
#SBATCH --partition=A40short
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=credit_alltok
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

# All-tokens re-extraction, CREDIT ONLY (first test: confirm 2D activations
# save correctly and check the storage hit before doing all four domains).
# Writes to artifacts/qwen35_4b_credit_alltokens/ (fresh dir, leaves your
# existing last-token credit data untouched).

source ~/NLP_Lab/setup_env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -euo pipefail
cd ~/NLP_Lab

echo "=== environment ==="
nvidia-smi || true
echo "=== position setting in config ==="
grep -i "position" configs/exp_qwen35_4b_credit_alltokens.yaml || true

echo "=== extracting credit with all_tokens activations ==="
python scripts/01_extract.py --config configs/exp_qwen35_4b_credit_alltokens.yaml

echo "=== done. Verify 2D with: ==="
echo "python run_all_tokens.py artifacts/qwen35_4b_credit_alltokens/activations"