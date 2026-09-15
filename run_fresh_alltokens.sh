#!/bin/bash
#SBATCH --partition=A100short
#SBATCH --time=8:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --export=NONE
#SBATCH --job-name=fresh_alltok
#SBATCH --output=logs/fresh_alltok_%j.out
#SBATCH --error=logs/fresh_alltok_%j.err
#
# Fresh all-tokens run of Qwen3.5-4B through the CURRENT pipeline, so the
# headline artifacts have a config that describes them and can be reproduced.
# Writes to artifacts/fresh_qwen35_4b_pooled_alltokens/ -- the original is
# left intact for comparison. Ends with a document-by-document diff against
# it, so the labeling drift is visible field by field.
#
# Text comes from data/parsed/pymupdf/ (the verified-deterministic cache).
# position=all stores every token's activation: expect ~1.2GB.

unset SLURM_EXPORT_ENV
source ~/NLP_Lab/setup_env_a100.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
set -euo pipefail
cd ~/NLP_Lab

CFG=configs/exp_fresh_alltokens.yaml
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "=== [1/2] extract ==="
python scripts/01_extract.py --config "$CFG"
echo "=== [2/2] label ==="
python scripts/02_label.py --config "$CFG"

echo ""
echo "=== fresh vs original alltokens, document by document ==="
python - << 'PYEOF'
import json, glob
from pathlib import Path

def load(d):
    out = {}
    for f in sorted(glob.glob(f"artifacts/{d}/labels/*.json")):
        if f.endswith("_summary.json"): continue
        x = json.load(open(f))
        L = [l for l in x.get("labels", []) if l.get("extracted_present", True)]
        out[x.get("doc_id", Path(f).stem)] = (len(L), sum(int(l.get("is_error",0)) for l in L))
    return out

new = load("fresh_qwen35_4b_pooled_alltokens")
old = load("qwen35_4b_pooled_alltokens")
print(f"{'document':50} {'fresh':>12} {'original':>12}")
same = 0
for k in sorted(set(new) | set(old)):
    a = f"{new[k][0]:4}/{new[k][1]:4}" if k in new else "     ---"
    b = f"{old[k][0]:4}/{old[k][1]:4}" if k in old else "     ---"
    eq = k in new and k in old and new[k] == old[k]
    same += eq
    print(f"{k[:50]:50} {a:>12} {b:>12}{'' if eq else '  <--'}")
print(f"\n{same} identical of {len(set(new)|set(old))}")
print("fields/errors, extracted_present only. Differences with identical")
print("field counts are labeler/matcher drift; differing field counts point")
print("at the extraction or schema handling.")
PYEOF