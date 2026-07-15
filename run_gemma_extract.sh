#!/bin/bash
#SBATCH --partition=A100short
#SBATCH --time=8:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --export=NONE
#SBATCH --job-name=gemma_extract
#SBATCH --output=logs/gemma_extract_%j.out
#SBATCH --error=logs/gemma_extract_%j.err
# Gemma-3-12B full extraction on ExtractBench (A100 -- 12B + activations OOM'd A40).
# Full pipeline: extract -> label -> nested LODO (10kq already excluded via domains).
unset SLURM_EXPORT_ENV
source ~/NLP_Lab/setup_env_a100.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
set -euo pipefail
cd ~/NLP_Lab
echo "=== gpu ==="; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "=== [1/3] extract ==="
python scripts/01_extract.py --config configs/exp_gemma3_12b_pooled.yaml

echo "=== [2/3] label ==="
python scripts/02_label.py --config configs/exp_gemma3_12b_pooled.yaml

echo "=== [3/3] nested LODO (unbiased layer selection) ==="
python scripts/05b_nested_lodo.py --config configs/exp_gemma3_12b_pooled.yaml --out-name nested_lodo.json

echo "=== done. results in artifacts/gemma3_12b_pooled/results/ ==="