#!/bin/bash
#SBATCH --partition=A40short
#SBATCH --gpus=1
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --job-name=sob_full
#SBATCH --output=logs/sob_full_%j.out
#SBATCH --error=logs/sob_full_%j.err
# Qwen3.5-4B on the first 1000 SOB test records, then labeling.
# Smoke run: ~4.4 s/record, so ~75 min expected. Resubmit if the time limit hits.
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
source ~/NLP_Lab/setup_env.sh
set -euo pipefail
cd ~/NLP_Lab

# Use --resume only if this repo's 01_extract.py supports it.
RESUME=""
if grep -q -- "--resume" scripts/01_extract.py; then
  RESUME="--resume"
  echo "resume supported: skipping records already extracted"
else
  echo "no --resume in 01_extract.py: extracting all 1000"
fi

python scripts/01_extract.py --config configs/exp_qwen35_4b_sob_1k.yaml $RESUME
python scripts/02_label.py --config configs/exp_qwen35_4b_sob_1k.yaml