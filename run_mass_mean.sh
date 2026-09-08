#!/bin/bash
#SBATCH --partition=A100short
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --export=NONE
#SBATCH --job-name=mass_mean
#SBATCH --output=logs/mass_mean_%j.out
#SBATCH --error=logs/mass_mean_%j.err
#
# Mass-mean probe vs logistic regression under identical nested LODO.
#
# NO GPU is requested -- this is a CPU job. It runs on an A100 node only
# because Bender has no CPU-only partition and A40 is contended.
# Bender requires --export=NONE to be paired with `unset SLURM_EXPORT_ENV`,
# and A100 nodes need their own environment build.
#
# Thread caps matter here: the nested loop fits many models and sklearn will
# otherwise grab every core on the node.

unset SLURM_EXPORT_ENV
source ~/NLP_Lab/setup_env_a100.sh

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

set -euo pipefail
cd ~/NLP_Lab

python mass_mean_probe.py \
    --domain qwen35_4b_pooled_alltokens \
    --layers 14 16 18 20 22 \
    --exclude-domains finance/10kq