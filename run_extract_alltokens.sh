#!/bin/bash
#SBATCH --partition=A40short
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=extract_alltok
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

# Re-extraction with position: all_tokens (saves every token's activation per
# field) so we can compute last / mean / span-max downstream from one run.
#
# PREREQUISITE: set  position: "all_tokens"  in the config's activations block
# BEFORE submitting. The all-tokens config produces larger .npz files (~mean
# span length x bigger), hence --mem=64G and a roomier time limit.

source ~/NLP_Lab/setup_env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -euo pipefail
cd ~/NLP_Lab

echo "=== environment ==="
nvidia-smi || true
grep -i "position" configs/exp_qwen35_4b_pooled_alltokens.yaml || true
echo "=== re-extracting pooled domains with all_tokens activations ==="

# NOTE: extraction is per-domain. The pooled set is built from the four
# per-domain extractions, so re-extract each domain that feeds the pool.
for cfg in pymupdf 10kq credit swimming; do
  echo "--- extracting $cfg (all_tokens) ---"
  python scripts/01_extract.py \
    --config configs/exp_qwen35_4b_${cfg}_alltokens.yaml \
    || echo "=== $cfg FAILED, continuing ==="
done

echo "=== done; now (CPU) rebuild pool + run 06_span_aggregation ==="