#!/bin/bash
#SBATCH --partition=A40short
#SBATCH --time=0:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --job-name=reproduce_44b
#SBATCH --output=logs/reproduce44b_%j.out
#SBATCH --error=logs/reproduce44b_%j.err
# Reproduce +44 with the CORRECT probe (qwen35_4b_pooled, per the docstring),
# trying BOTH caches. Previous attempt used the wrong probe dir (pymupdf).
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
source ~/NLP_Lab/setup_env.sh
set -euo pipefail
cd ~/NLP_Lab

PROBE=artifacts/qwen35_4b_pooled/probes/probe_layer18.pkl

echo "############## CACHE v2 + correct probe ##############"
python scripts/safe_override_rescore.py \
    --domain qwen35_4b_pooled_alltokens \
    --cache artifacts/qwen35_4b_pooled_alltokens/results/regen_cache_pooled_v2.json \
    --probe-path $PROBE --layer 18 | tee logs/plus44_v2_correctprobe.txt

echo ""
echo "############## CACHE v1 + correct probe ##############"
python scripts/safe_override_rescore.py \
    --domain qwen35_4b_pooled_alltokens \
    --cache artifacts/qwen35_4b_pooled_alltokens/results/regen_cache_pooled.json \
    --probe-path $PROBE --layer 18 | tee logs/plus44_v1_correctprobe.txt