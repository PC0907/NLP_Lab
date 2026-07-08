#!/bin/bash
#SBATCH --partition=A40short
#SBATCH --time=0:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --job-name=transfer_probe
#SBATCH --output=logs/transfer_%j.out
#SBATCH --error=logs/transfer_%j.err
# CPU-only: cross-dataset transfer (ExtractBench probe -> insurance). No GPU.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
source ~/NLP_Lab/setup_env.sh
set -euo pipefail
cd ~/NLP_Lab

python transfer_probe.py \
    --train-config configs/exp_qwen35_4b_pooled_alltokens.yaml \
    --test-config  configs/exp_qwen35_4b_insurance.yaml \
    --layer 16 \
    --train-exclude-domains finance/10kq \
    --out transfer_extractbench_to_insurance.json