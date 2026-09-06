#!/bin/bash
#SBATCH --partition=A40short
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --job-name=nested_regen
#SBATCH --output=logs/nested_regen_%j.out
#SBATCH --error=logs/nested_regen_%j.err
#
# Fully nested safe-override: both layer and threshold selected on inner
# documents. CPU only, no GPU requested. Thread caps matter -- the inner loop
# fits many logistic regressions and sklearn will otherwise take every core.

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

source ~/NLP_Lab/setup_env.sh
set -euo pipefail
cd ~/NLP_Lab

python nested_safe_override.py \
    --domain qwen35_4b_pooled_alltokens \
    --cache artifacts/qwen35_4b_pooled_alltokens/results/regen_cache_pooled_v2.json \
    --layers 14 16 18 20 22 \
    --objective both