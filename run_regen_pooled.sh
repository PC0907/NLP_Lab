#!/bin/bash
#SBATCH --partition=A100short
#SBATCH --export=NONE
#SBATCH --time=3:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --job-name=regen_pooled
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

# Pooled selective regeneration (all four domains, one probe, one prompt, one
# scoring rule -> the generalization test). Writes a fresh pooled cache.
#
# PREREQUISITES (run on login node first, both CPU):
#   1. fixability on pooled:
#        python scripts/08_fixability_filter.py --config configs/exp_qwen35_4b_pooled_regen.yaml
#      -> writes artifacts/qwen35_4b_pooled_alltokens/results/fixability.json
#   2. confirm a probe exists for scoring (the pooled probe), and the corrected
#      09_regen_sweep.py (with list_info / position_hint) is in place.

source ~/NLP_Lab/setup_env_a100.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -euo pipefail
cd ~/NLP_Lab

echo "=== environment ==="
nvidia-smi || true
echo "=== prompt fix present? (expect >0) ==="
grep -c "list_info\|position_hint" scripts/09_regen_sweep.py || echo "WARNING: fix missing"
echo "=== fixability present? ==="
ls -la artifacts/qwen35_4b_pooled_alltokens/results/fixability.json || echo "WARNING: run 08 first"

echo "=== regenerate pooled (all four domains) -> fresh cache ==="
python scripts/09_regen_sweep.py \
  --config configs/exp_qwen35_4b_pooled_regen.yaml \
  --probe-path artifacts/qwen35_4b_pooled/probes/probe_layer18.pkl \
  --regenerate \
  --temperature 0.0 \
  --cache artifacts/qwen35_4b_pooled_alltokens/results/regen_cache_pooled.json

echo "=== done. Next (CPU): rescore_regen on the pooled cache. ==="