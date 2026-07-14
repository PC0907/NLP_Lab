#!/bin/bash
#SBATCH --partition=A40short
#SBATCH --time=1:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=smoke_gemma
#SBATCH --output=logs/smoke_gemma_%j.out
#SBATCH --error=logs/smoke_gemma_%j.err
# Gemma schema-key smoke test: ONE doc, does Gemma emit literal schema keys?
# The Llama-risk gate. Gemma-3-12B bf16 ~24GB -> fits an A40 (46GB).
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
source ~/NLP_Lab/setup_env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -euo pipefail
cd ~/NLP_Lab
echo "=== gpu ==="; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Test on an ACADEMIC doc first (nested schema -- the kind that broke Llama).
python smoke_test_gemma.py --config configs/exp_gemma4_12b_pooled.yaml --doc-index 0