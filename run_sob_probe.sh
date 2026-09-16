#!/bin/bash
#SBATCH --partition=A40short
#SBATCH --time=6:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --job-name=sob_probe
#SBATCH --output=logs/sob_probe_%j.out
#SBATCH --error=logs/sob_probe_%j.err
#
# SOB (997 records): data checks, nested grouped 5x5 K-fold, fixed-layer sweep,
# final probes. CPU only, no GPU requested. Report -> .out, progress -> .err.

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

source ~/NLP_Lab/setup_env.sh
set -euo pipefail
cd ~/NLP_Lab

python scripts/05c_nested_groupkfold.py \
    --config configs/exp_qwen35_4b_sob_1k.yaml \
    --layers 14 16 18 20 22