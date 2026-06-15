#!/bin/bash
#SBATCH --partition=A100short
#SBATCH --export=NONE
#SBATCH --time=2:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --job-name=regen_acad_fix
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

# Validate the list-element prompt fix: regenerate academic candidates with the
# corrected correction-with-context prompt, writing to a FRESH cache (so the old
# academic cache is preserved for before/after comparison and the resume logic
# does not skip everything).
#
# PREREQ: the corrected scripts/09_regen_sweep.py is in place
#         (grep -c "list_info" scripts/09_regen_sweep.py  -> should be > 0).

source ~/NLP_Lab/setup_env_a100.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -euo pipefail
cd ~/NLP_Lab

echo "=== environment ==="
nvidia-smi || true
echo "=== confirm prompt fix is present (expect a number > 0) ==="
grep -c "list_info\|position_hint" scripts/09_regen_sweep.py || echo "WARNING: fix may not be applied"
echo "=== confirm probe exists ==="
ls -la artifacts/qwen35_4b_pooled/probes/ || true

echo "=== regenerate academic with fixed prompt -> FRESH cache ==="
python scripts/09_regen_sweep.py \
  --config configs/exp_qwen35_4b_pymupdf.yaml \
  --probe-path artifacts/qwen35_4b_pooled/probes/probe_layer18.pkl \
  --regenerate \
  --temperature 0.0 \
  --cache artifacts/qwen35_4b_pymupdf/results/regen_cache_fixed.json

echo "=== done. Next (CPU): rescore + inspect authors.N.name from the new cache ==="
echo "  cache: artifacts/qwen35_4b_pymupdf/results/regen_cache_fixed.json"