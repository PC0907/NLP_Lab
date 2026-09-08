#!/bin/bash
#SBATCH --job-name=nested_repro
#SBATCH --partition=A40devel
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/nested_repro_%j.out
#SBATCH --error=logs/nested_repro_%j.err

set -euo pipefail
cd "$HOME/NLP_Lab"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate nlp_lab

mkdir -p logs

python scripts/05b_nested_lodo.py \
  --config configs/exp_qwen35_4b_pooled_alltokens.yaml \
  --layers 14 16 18 20 22 \
  --out-name nested_lodo_repro.json