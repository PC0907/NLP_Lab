#!/bin/bash
#SBATCH --partition=A40short
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --job-name=mass_mean
#SBATCH --output=logs/mass_mean_%j.out
#SBATCH --error=logs/mass_mean_%j.err
#
# Mass-mean probe vs logistic regression under identical nested LODO.
# CPU only, no GPU requested. Thread caps matter: the nested loop fits many
# models and sklearn will otherwise take every core on the node.

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

source ~/NLP_Lab/setup_env.sh
set -euo pipefail
cd ~/NLP_Lab

python mass_mean_probe.py \
    --domain qwen35_4b_pooled_alltokens \
    --layers 14 16 18 20 22 \
    --exclude-domains finance/10kq