#!/bin/bash
#SBATCH --partition=A40short
#SBATCH --gpus=1
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --job-name=sob_smoke
#SBATCH --output=logs/sob_smoke_%j.out
#SBATCH --error=logs/sob_smoke_%j.err
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
source ~/NLP_Lab/setup_env.sh
set -euo pipefail
cd ~/NLP_Lab
python scripts/01_extract.py --config configs/exp_qwen35_4b_sob_1k.yaml --limit 20
python scripts/02_label.py --config configs/exp_qwen35_4b_sob_1k.yaml