#!/bin/bash
#SBATCH --partition=A100short
#SBATCH --export=NONE
#SBATCH --time=8:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=qwen_alltok_a100
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

# All-tokens re-extraction on A100 (80GB) for all four Qwen domains, so we can
# build a pooled all-tokens set and compare last/mean/span-max on the full
# distribution. A100 avoids the A40 OOM on long credit/10kq docs (no truncation
# needed -> full document text preserved).
#
# PREREQUISITES on the cluster before submitting:
#   - extractor.py has position "all" in SUPPORTED_POSITIONS
#   - the four configs/exp_qwen35_4b_{pymupdf,10kq,credit,swimming}_alltokens.yaml exist
#     (position: "all", fresh artifact names ..._alltokens)

source ~/NLP_Lab/setup_env_a100.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -euo pipefail
cd ~/NLP_Lab

echo "=== environment ==="
echo "which python: $(which python || echo 'not found')"
echo "python version: $(python --version 2>&1 || echo 'failed')"
nvidia-smi || true
echo "=== environment OK ==="

# swimming/10kq first (shorter), then the long ones (pymupdf/credit)
for cfg in swimming 10kq pymupdf credit; do
  echo "=== extracting $cfg (all_tokens) ==="
  grep -i "position" configs/exp_qwen35_4b_${cfg}_alltokens.yaml || true
  python scripts/01_extract.py --config configs/exp_qwen35_4b_${cfg}_alltokens.yaml \
    || echo "=== $cfg FAILED, continuing ==="
done

echo "=== all four domains done ==="
echo "Next (CPU): label each, rebuild pool, run 06_span_aggregation."