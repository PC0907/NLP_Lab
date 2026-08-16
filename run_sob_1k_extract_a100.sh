#!/bin/bash
#SBATCH --partition=A100medium
#SBATCH --export=NONE
#SBATCH --time=12:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=sob_1k_extract
#SBATCH --output=logs/sob_1k_extract-%j.out
#SBATCH --error=logs/sob_1k_extract-%j.err

# Scale the SOB attribution corpus 300 -> 1,000 documents, to turn a borderline
# p=0.044 into a result that survives multiple-comparison correction.
#
# RESUMABLE AND SHARDABLE. Two things make this safe to run against a wall clock:
#   --resume  skips documents already extracted WITH per-token reasoning states,
#             so the ~300 from the first attribution run cost nothing, and a
#             time-limit kill loses only the document in flight. Just resubmit.
#   --shard   optional: run this script several times with SHARD=1/3, 2/3, 3/3
#             to split the corpus across concurrent GPU jobs (disjoint files).
#
# Usage:
#   sbatch run_sob_1k_extract_a100.sh              # one job, resume-safe
#   SHARD=1/3 sbatch run_sob_1k_extract_a100.sh    # or three concurrent jobs
#   (resubmit the same command after a time-limit kill; it picks up where it left off)

module load Python/3.12.3
module load CUDA/12.4.0
source ~/nlp_lab_a100/bin/activate
export PYTHONPATH=$HOME/NLP_Lab/src:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_DATASETS_OFFLINE=1

# MUST match the first attribution run: Stage 7 requires every requested layer
# to be present in every document, so a narrower set here would strand the
# documents already on disk.
export REASONING_TOKEN_LAYERS="16,19,23,26"
export REASONING_TOKEN_CAP="2048"

set -euo pipefail
cd ~/NLP_Lab

CFG="configs/exp_deepseek_r1_7b_sob_attr_1k.yaml"
ART="artifacts/deepseek_r1_7b_sob_attr"
SHARD_ARG=""
if [ -n "${SHARD:-}" ]; then SHARD_ARG="--shard ${SHARD}"; fi

echo "=== ENVIRONMENT ==="
hostname; nvidia-smi || true
echo "already extracted with reasoning tokens: $(ls ${ART}/activations/*.rtokens.json 2>/dev/null | wc -l)"

# Disk guard. Per-token reasoning states run ~50 MB/doc; 700 new documents need
# roughly 35 GB. Better to refuse now than to die at document 600.
AVAIL_GB=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc '0-9')
echo "free space in \$HOME: ${AVAIL_GB} GB"
if [ "${AVAIL_GB:-0}" -lt 40 ]; then
  echo "FATAL: need ~40 GB free for the new documents, have ${AVAIL_GB} GB."
  echo "       Free space, or point experiment.artifacts_dir at a larger filesystem."
  exit 1
fi

# Real GPU self-test -- abort before any silent CPU fallback.
python - <<'PY' || { echo "FATAL: GPU not usable; resubmit with --exclude=$(hostname -s)."; exit 1; }
import torch, sys
try:
    assert torch.cuda.is_available() and torch.cuda.device_count() > 0
    x = torch.zeros(2048, 2048, device="cuda:0"); _ = (x + 1.0).sum().item()
    free, total = torch.cuda.mem_get_info(0)
    print(f"GPU OK | {torch.cuda.get_device_name(0)} | free {free/1e9:.1f}/{total/1e9:.1f} GB")
except Exception as e:
    print("GPU SELF-TEST FAILED:", repr(e)); sys.exit(1)
PY
echo "=== GPU GUARD PASSED ==="

echo "=== STAGE 01: extraction to 1,000 docs (resume) ${SHARD_ARG} ==="
python scripts/01_extract.py --config "$CFG" --resume ${SHARD_ARG}

echo "=== STAGE 01 COMPLETE ==="
echo "docs with reasoning tokens now: $(ls ${ART}/activations/*.rtokens.json 2>/dev/null | wc -l)"
du -sh ${ART}/activations || true
echo "Next: sbatch run_sob_1k_analysis_a100.sh"
