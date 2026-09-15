#!/bin/bash
#SBATCH --partition=A100short
#SBATCH --time=8:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --export=NONE
#SBATCH --job-name=trunc_probe
#SBATCH --output=logs/trunc_probe_%j.out
#SBATCH --error=logs/trunc_probe_%j.err
# CPU only. Intersection + nested LODO on both truncated cells.
# --layers restricted to the mid-network band: the full 14-layer band timed
# out on the grid run, and nested detection already showed the signal lives
# at 14-22.

unset SLURM_EXPORT_ENV
source ~/NLP_Lab/setup_env_a100.sh
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
set -euo pipefail
cd ~/NLP_Lab

# grid_probe.py looks for artifacts/grid_<model>_<parser>; symlink the
# truncated cells under a parser name it will pick up.
mkdir -p artifacts
ln -sfn "$PWD/artifacts/trunc_qwen35_4b_pymupdf"  artifacts/grid_qwen35_4b_trunc300k
ln -sfn "$PWD/artifacts/trunc_gemma3_4b_pymupdf"  artifacts/grid_gemma3_4b_trunc300k
ln -sfn "$PWD/configs/exp_trunc_qwen35_4b_pymupdf.yaml"  configs/exp_grid_qwen35_4b_trunc300k.yaml
ln -sfn "$PWD/configs/exp_trunc_gemma3_4b_pymupdf.yaml"  configs/exp_grid_gemma3_4b_trunc300k.yaml

python grid_probe.py \
    --parsers trunc300k \
    --models qwen35_4b gemma3_4b \
    --exclude-domains finance/10kq \
    --layers 14 16 18 20 22 \
    --out artifacts/trunc_probe_comparison.json
