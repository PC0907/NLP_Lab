#!/bin/bash
#SBATCH --partition=A100short
#SBATCH --export=NONE
#SBATCH --time=8:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --job-name=regen_pooled_v2
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

# Pooled selective regeneration, v2: ENRICHED correction prompt
#   - injects each field's schema description/examples (fixes unit->millions)
#   - keeps list-element siblings populated as an identity anchor (fixes the
#     authors.N -> first-author default)
#   - MAX_DOC_CHARS truncation (no context overflow)
# Writes a FRESH v2 cache so break counts can be compared to v1 (broke=354 lenient).
#
# PREREQS (verify on login node before submitting):
#   grep -c "_build_schema_block" scripts/09_regen_sweep.py   # expect >0 (enriched)
#   grep -c "MAX_DOC_CHARS"        scripts/09_regen_sweep.py   # expect >0 (truncation)
#   ls artifacts/qwen35_4b_pooled/probes/probe_layer18.pkl
#   ls artifacts/qwen35_4b_pooled_alltokens/results/fixability.json

source ~/NLP_Lab/setup_env_a100.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -euo pipefail
cd ~/NLP_Lab

echo "=== environment ==="
nvidia-smi || true
echo "=== enriched prompt present? (expect >0 each) ==="
grep -c "_build_schema_block" scripts/09_regen_sweep.py || echo "WARN: schema block missing"
grep -c "MAX_DOC_CHARS"        scripts/09_regen_sweep.py || echo "WARN: truncation missing"
echo "=== probe present? ==="
ls -la artifacts/qwen35_4b_pooled/probes/probe_layer18.pkl || echo "WARN: probe missing"

echo "=== pooled regeneration v2 (enriched prompt) -> FRESH v2 cache ==="
python scripts/09_regen_sweep.py \
  --config configs/exp_qwen35_4b_pooled_regen.yaml \
  --probe-path artifacts/qwen35_4b_pooled/probes/probe_layer18.pkl \
  --regenerate \
  --temperature 0.0 \
  --cache artifacts/qwen35_4b_pooled_alltokens/results/regen_cache_pooled_v2.json

echo "=== done. Next (CPU): rescore + dump v2, compare break counts to v1 (354). ==="
echo "  rescore: python scripts/rescore_regen.py --domain qwen35_4b_pooled_alltokens \\"
echo "           --cache artifacts/qwen35_4b_pooled_alltokens/results/regen_cache_pooled_v2.json \\"
echo "           --probe-path artifacts/qwen35_4b_pooled/probes/probe_layer18.pkl"