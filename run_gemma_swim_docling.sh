#!/bin/bash
#SBATCH --partition=A100short
#SBATCH --time=2:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --export=NONE
#SBATCH --job-name=gem_swim_dl
#SBATCH --output=logs/gem_swim_dl_%j.out
#SBATCH --error=logs/gem_swim_dl_%j.err
#
# Gemma-3-4B on the five swimming documents, parsed with Docling.
#
# ONE HYPOTHESIS: Gemma-3-4B's swimming failure under pymupdf (15-86% error,
# vs 0-13% for the 12B on identical documents) was caused by the parser
# flattening tables to one value per line with no row delimiters, not by a
# capacity limit. Docling preserves table structure, so rows should arrive
# intact. Five short documents, so this answers the question cheaply before
# committing to a four-domain run.
#
# Docling is ~10x slower than pymupdf, so parsing dominates here, not generation.
# Gemma-3-4B will re-download (~8GB) -- it was cleared from the HF cache.

unset SLURM_EXPORT_ENV
source ~/NLP_Lab/setup_env_a100.sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

set -euo pipefail
cd ~/NLP_Lab
echo "=== gpu ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=== cv2 (docling needs it) ==="
python -c "import cv2; print('cv2', cv2.__version__)"

echo "=== [1/2] extract (swimming, docling) ==="
python scripts/01_extract.py --config configs/exp_gemma3_4b_swim_docling.yaml

echo "=== [2/2] label ==="
python scripts/02_label.py --config configs/exp_gemma3_4b_swim_docling.yaml

echo ""
echo "=== VERDICT ==="
python -c "
import json, glob

def per_doc(d):
    out = {}
    for f in glob.glob(f'artifacts/{d}/labels/*.json'):
        if f.endswith('_summary.json'):
            continue
        x = json.load(open(f))
        L = x.get('labels', [])
        if not L:
            continue
        doc = f.split('/')[-1].replace('.json', '')
        n = len(L)
        e = sum(int(i.get('is_error', 0)) for i in L)
        out[doc] = (n, e)
    return out

new = per_doc('gemma3_4b_swim_docling')
old = per_doc('gemma3_4b_pooled')   # pymupdf run, swimming subset

print(f'{\"document\":42} {\"docling\":>18}   {\"pymupdf (was)\":>18}')
for doc in sorted(new):
    n, e = new[doc]
    cur = f'{n:4} fields {e/max(n,1):5.1%}'
    if doc in old:
        on, oe = old[doc]
        prev = f'{on:4} fields {oe/max(on,1):5.1%}'
    else:
        prev = '               ---'
    print(f'{doc[:42]:42} {cur:>18}   {prev:>18}')

tn = sum(n for n, _ in new.values())
te = sum(e for _, e in new.values())
print()
print(f'docling swimming overall: {tn} fields, {te} errors ({te/max(tn,1):.1%})')
print()
print('GATE: pymupdf gave 15-86% per document (58% overall).')
print('      Gemma-3-12B managed 0-13% on the same documents under pymupdf.')
print('      If docling lands near the 12B range, the failure was the PARSER')
print('      and a four-domain run is worth committing to.')
print('      If it stays high, the failure is CAPACITY and no parser will fix it.')
"