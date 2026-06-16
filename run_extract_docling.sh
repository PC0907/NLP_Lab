#!/bin/bash
#SBATCH --partition=A100short
#SBATCH --export=NONE
#SBATCH --time=8:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=docling_extract
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

# Full Docling re-extraction, all four domains. NOTE: 01_extract does BOTH the
# PDF->text (Docling backend, CPU, slow) AND the Qwen LLM generation with
# activation capture (GPU). So this needs a GPU even though Docling itself is
# CPU. Docling's deep layout/table parsing is much slower than PyMuPDF (minutes
# per doc), hence the long time limit.
#
# Writes to artifacts/qwen35_4b_{domain}_docling/ (fresh dirs; leaves PyMuPDF
# data intact). Last-token activations (default) -- this run is about PARSING
# QUALITY (gold-coverage), not span-max, so we don't need all-tokens here.

source ~/NLP_Lab/setup_env_a100.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -euo pipefail
cd ~/NLP_Lab

echo "=== environment ==="
nvidia-smi || true
python -c "import docling; print('docling import OK')" || echo "WARNING: docling import failed"

# shortest domains first so we see results / catch errors early
for cfg in swimming pymupdf 10kq credit; do
  echo "=== Docling extract: $cfg ==="
  grep -i "pdf_extractor" configs/exp_qwen35_4b_${cfg}_docling.yaml || true
  python scripts/01_extract.py --config configs/exp_qwen35_4b_${cfg}_docling.yaml \
    || echo "=== $cfg FAILED, continuing ==="
done

echo "=== extraction done. Next (CPU): label each, then 08b_gold_coverage to"
echo "    compare Docling vs PyMuPDF gold-in-text rates. ==="