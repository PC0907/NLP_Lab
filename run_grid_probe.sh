#!/bin/bash
#SBATCH --partition=A100short
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --export=NONE
#SBATCH --job-name=grid_probe
#SBATCH --output=logs/grid_probe_%j.out
#SBATCH --error=logs/grid_probe_%j.err
# CPU only, no GPU. Nested LODO on each (model, parser) cell, restricted to
# the documents both models completed for that parser.

unset SLURM_EXPORT_ENV
source ~/NLP_Lab/setup_env_a100.sh
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
set -euo pipefail
cd ~/NLP_Lab

python grid_probe.py \
    --parsers pymupdf docling camelot \
    --models qwen35_4b gemma3_4b \
    --exclude-domains finance/10kq