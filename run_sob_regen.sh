#!/bin/bash
#SBATCH --partition=A40short
#SBATCH --gpus=1
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --job-name=sob_regen
#SBATCH --output=logs/sob_regen_%j.out
#SBATCH --error=logs/sob_regen_%j.err
# SOB: fixability filter, then GPU regeneration of all eligible scalar fields.
# Resumable: resubmit if the time limit hits; cached fields are skipped.
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
source ~/NLP_Lab/setup_env.sh
set -euo pipefail
cd ~/NLP_Lab
CFG=configs/exp_qwen35_4b_sob_1k.yaml
A=artifacts/qwen35_4b_sob_1k

if [ ! -f $A/results/fixability.json ]; then
  python scripts/08_fixability_filter.py --config $CFG
fi
python scripts/09_regen_sweep.py --config $CFG \
    --probe-path $A/probes/probe_layer18.pkl --layer 18 --regenerate