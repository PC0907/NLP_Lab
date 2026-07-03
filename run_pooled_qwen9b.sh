#!/bin/bash
#SBATCH --partition=A100short
#SBATCH --time=8:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --export=NONE
#SBATCH --job-name=pooled_qwen9b
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
# Qwen3.5-9B pooled extraction. A100 (9B + hidden-4096 + long docs).
# Bender: --export=NONE needs the companion unset (per HPC wiki Job Environment
# section) so the env propagates to srun steps; without it the job can fail
# confusingly. A100 nodes are AMD -> need the A100-built env.
unset SLURM_EXPORT_ENV
source ~/NLP_Lab/setup_env_a100.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -euo pipefail
cd ~/NLP_Lab
echo "=== node/gpu ==="; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=== [1/3] extract (GPU) ==="
python scripts/01_extract.py --config configs/exp_qwen35_9b_pooled.yaml
echo "=== [2/3] label (CPU) ==="
python scripts/02_label.py --config configs/exp_qwen35_9b_pooled.yaml
echo "=== [3/3] nested LODO, 10kq excluded (CPU) ==="
python scripts/05b_nested_lodo.py --config configs/exp_qwen35_9b_pooled.yaml --exclude-domains finance/10kq --out-name nested_lodo_nofin.json
echo "=== done. results in artifacts/qwen35_9b_pooled/results/ ==="