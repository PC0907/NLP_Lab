#!/bin/bash
#SBATCH --partition=A100medium
#SBATCH --export=NONE
#SBATCH --time=12:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=sob_regen
#SBATCH --output=logs/sob_regen-%j.out
#SBATCH --error=logs/sob_regen-%j.err

# STAGE 11 -- REAL regeneration (GPU).
#
# Everything the paper says about selective regeneration currently rests on two
# invented numbers: a re-asked wrong field is repaired with probability 0.7, a
# re-asked right field is broken with probability 0.05. This job replaces the
# assumption with measurement -- it actually re-asks the model, three times per
# document at temperature 0.7, and saves what it said. Stage 12 then checks what
# really happened against gold.
#
# Greedy decoding is NOT usable here: a temperature-0 re-run reproduces the
# original extraction token for token and could never repair anything. The
# script refuses to start if the temperature is 0.
#
# COST. ~1,000 documents x 3 samples, at roughly the per-document cost of the
# original extraction. Budget ~13 h of GPU time -- more than the 12 h wall clock,
# so run it as TWO SHARDS side by side (~6.5 h each):
#
#   sbatch run_sob_regenerate_a100.sh 1/2
#   sbatch run_sob_regenerate_a100.sh 2/2
#
# Shards are 1-based: I/N with I in 1..N.
#
# Shards write disjoint files, so they never collide. Or run a single job and
# resubmit after a time-limit kill -- --resume skips documents already done and
# loses only the one in flight:
#
#   sbatch run_sob_regenerate_a100.sh
#
# NOTE the positional argument. Passing the shard through an environment
# variable does NOT work: #SBATCH --export=NONE strips the environment, so the
# value never reaches the job. sbatch does forward positional arguments.

module load Python/3.12.3
module load CUDA/12.4.0
source ~/nlp_lab_a100/bin/activate
export PYTHONPATH=$HOME/NLP_Lab/src:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_DATASETS_OFFLINE=1

set -euo pipefail
cd ~/NLP_Lab

CFG="configs/exp_deepseek_r1_7b_sob_attr_1k.yaml"
ART="artifacts/deepseek_r1_7b_sob_attr"
SAMPLES=3

SHARD_ARG=""
if [ -n "${1:-}" ]; then SHARD_ARG="--shard ${1}"; fi

echo "=== ENVIRONMENT ==="
hostname; nvidia-smi || true
echo "extractions on disk : $(ls ${ART}/extractions/*.json 2>/dev/null | grep -cv '_summary' || true)"
echo "already regenerated : $(ls ${ART}/regen/*.json 2>/dev/null | wc -l)"

# Real GPU self-test -- abort before any silent CPU fallback turns 13 h into
# three weeks.
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

echo ""
echo "=== STAGE 11: regenerate ${SAMPLES} samples/document ${SHARD_ARG} ==="
python scripts/11_regenerate.py --config "$CFG" \
    --samples "$SAMPLES" --resume ${SHARD_ARG}

echo ""
echo "=== STAGE 11 COMPLETE ==="
echo "regenerated documents: $(ls ${ART}/regen/*.json 2>/dev/null | wc -l)"
du -sh ${ART}/regen || true

python - <<'PY'
import json, pathlib
d = pathlib.Path("artifacts/deepseek_r1_7b_sob_attr/regen")
files = sorted(d.glob("*.json"))
tot = trunc = parse = ident = 0
for f in files:
    p = json.loads(f.read_text())
    for s in p.get("samples", []):
        tot += 1
        if s.get("finish_reason") == "length":
            trunc += 1
        if s.get("parse_error"):
            parse += 1
print(f"samples: {tot} across {len(files)} documents")
if tot:
    print(f"  hit max_new_tokens : {trunc} ({100*trunc/tot:.1f}%)  -- excluded by Stage 12")
    print(f"  parse failures     : {parse} ({100*parse/tot:.1f}%)")
    if trunc / tot > 0.25:
        print("  WARNING: a quarter or more of the resamples were cut off. Their JSON")
        print("           is a parser reconstruction, so Stage 12 drops them and the")
        print("           measurement loses that much coverage. Consider raising")
        print("           model.max_new_tokens before reading the result as final.")
PY

echo ""
echo "Next: sbatch run_sob_regen_eval.sh   (Stage 9 rescore + Stage 12 measurement, CPU)"
