#!/bin/bash
#SBATCH --partition=A100short
#SBATCH --time=1:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --export=NONE
#SBATCH --job-name=smoke_gemma
#SBATCH --output=logs/smoke_gemma_%j.out
#SBATCH --error=logs/smoke_gemma_%j.err
# Gemma-3-12B schema-key smoke test. A100 (80GB): the 12B at bf16 (~24GB) plus
# KV cache for a long doc plus hidden-state capture OOM'd on the A40 (44GB).
# Bender: --export=NONE needs the companion unset (HPC wiki), and A100 nodes are
# AMD -> need the A100-built env.
unset SLURM_EXPORT_ENV
source ~/NLP_Lab/setup_env_a100.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
set -euo pipefail
cd ~/NLP_Lab
echo "=== gpu ==="; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Doc 0 is an academic paper (long). If this still OOMs, point --doc-index at a
# short swimming doc -- the smoke test only checks KEY STRUCTURE, so a short
# document answers the question just as well.
python smoke_test_gemma.py --config configs/exp_gemma3_12b_swim.yaml --doc-index 0