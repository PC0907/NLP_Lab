#!/bin/bash
#SBATCH --job-name=qwen9b_pooled
#SBATCH --partition=A100short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --export=NONE
#SBATCH --output=logs/qwen9b_pooled_%j.out
#SBATCH --error=logs/qwen9b_pooled_%j.err

# --- IMPORTANT --------------------------------------------------------------
# This preamble is RECONSTRUCTED from the project notes (A100short, --export=NONE,
# setup_env_a100.sh, expandable_segments). Before submitting, DIFF the block below
# against your known-working A100 script (the one that produced the 4B alltokens
# run). If that script sets modules / venv differently, prefer ITS preamble --
# do not trust this reconstruction over a script you've already run successfully.
# ----------------------------------------------------------------------------

set -euo pipefail

source setup_env_a100.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONFIG=configs/exp_qwen35_9b_pooled.yaml

echo "=== node / gpu ==="
hostname
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "=== [1/3] extract (GPU) ==="
python scripts/01_extract.py --config "$CONFIG"

echo "=== [2/3] label (CPU) ==="
python scripts/02_label.py --config "$CONFIG"

echo "=== [3/3] nested LODO, 10kq excluded (CPU) ==="
python scripts/05b_nested_lodo.py \
    --config "$CONFIG" \
    --exclude-domains finance/10kq \
    --out-name nested_lodo_nofin.json

echo "=== done. results in artifacts/qwen35_9b_pooled/results/ ==="