#!/bin/bash
#SBATCH --partition=A100short
#SBATCH --time=8:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --export=NONE
#SBATCH --job-name=gemma_s1
#SBATCH --output=logs/gemma_s1_%j.out
#SBATCH --error=logs/gemma_s1_%j.err
# STAGE 1: Gemma-3-12B extraction + labeling ONLY.
# Stops before the probe/LODO stage so the error rate can be inspected first.
# A100 required: 12B + activation capture OOMs on the A40.
unset SLURM_EXPORT_ENV
source ~/NLP_Lab/setup_env_a100.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
set -euo pipefail
cd ~/NLP_Lab
echo "=== gpu ==="; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "=== [1/2] extract (all documents) ==="
python scripts/01_extract.py --config configs/exp_gemma3_12b_pooled.yaml

echo "=== [2/2] label ==="
python scripts/02_label.py --config configs/exp_gemma3_12b_pooled.yaml

echo ""
echo "=== STAGE 1 COMPLETE -- inspect before running stage 2 ==="
python -c "
import json, glob
exts = [f for f in glob.glob('artifacts/gemma3_12b_pooled/extractions/*.json') if not f.endswith('_summary.json')]
ok = err_fin = 0
for f in exts:
    d = json.load(open(f))
    if d.get('finish_reason') == 'error': err_fin += 1
    elif d.get('parsed_json') is not None: ok += 1
print(f'extracted: {len(exts)} files | parsed ok: {ok} | failed: {err_fin}')

labs = [f for f in glob.glob('artifacts/gemma3_12b_pooled/labels/*.json') if not f.endswith('_summary.json')]
from collections import defaultdict
dom = defaultdict(lambda: [0,0])
tot = e = 0
for f in labs:
    d = json.load(open(f)); L = d.get('labels', [])
    dm = d.get('domain','?')
    n = len(L); ne = sum(int(x.get('is_error',0)) for x in L)
    dom[dm][0] += n; dom[dm][1] += ne
    tot += n; e += ne
print(f'labeled docs: {len(labs)} | fields: {tot} | errors: {e} ({e/max(tot,1):.1%})')
print('per-domain:')
for dm,(n,ne) in sorted(dom.items()):
    print(f'  {dm:32} fields={n:5} errors={ne:5} ({ne/max(n,1):.1%})')
print('')
print('GATE: compare to Qwen-4B (~11% on these domains). If Gemma is far higher,')
print('check whether it is genuine extraction error or a matcher/format mismatch')
print('BEFORE running the probe stage.')
"