#!/bin/bash
#SBATCH --partition=A100short
#SBATCH --time=1:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --export=NONE
#SBATCH --job-name=smoke_r1
#SBATCH --output=logs/smoke_r1_%j.out
#SBATCH --error=logs/smoke_r1_%j.err
#
# One-document smoke test for DeepSeek-R1-Distill-Qwen-7B.
# Checks: reasoning-trace length, whether the JSON completes in budget,
# whether the parser recovers it, and schema-key reproduction.
# 7B at bf16 is ~15GB; A100 for queue availability and headroom.

unset SLURM_EXPORT_ENV
source ~/NLP_Lab/setup_env_a100.sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

set -euo pipefail
cd ~/NLP_Lab
echo "=== gpu ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# doc-index 0 is an academic paper. If it truncates, retry on a shorter
# swimming document.
python smoke_test_r1.py --config configs/exp_r1qwen7b_pooled.yaml --doc-index 0