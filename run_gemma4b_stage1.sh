#!/bin/bash
#SBATCH --partition=A100short
#SBATCH --time=8:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=gemma4b_s1
#SBATCH --output=logs/gemma4b_s1_%j.out
#SBATCH --error=logs/gemma4b_s1_%j.err
#SBATCH --export=NONE
# STAGE 1: Gemma-3-4B extraction + labeling only. Stops before the probe stage
# so the error rate and document yield can be checked first.
# A40 (44GB) should suffice: 4B at bf16 is ~8GB of weights. If the longest
# credit agreements still OOM, move to A100short (and add the --export=NONE /
unset SLURM_EXPORT_ENV
source ~/NLP_Lab/setup_env_a100.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
set -euo pipefail
cd ~/NLP_Lab
echo "=== gpu ==="; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "=== [1/2] extract ==="
python scripts/01_extract.py --config configs/exp_gemma3_4b_pooled.yaml

echo "=== [2/2] label ==="
python scripts/02_label.py --config configs/exp_gemma3_4b_pooled.yaml

echo ""
echo "=== STAGE 1 SUMMARY -- check yield and error rate before stage 2 ==="
python -c "
import json, glob
from collections import defaultdict
exts = [f for f in glob.glob('artifacts/gemma3_4b_pooled/extractions/*.json') if not f.endswith('_summary.json')]
ok = bad = 0
for f in exts:
    d = json.load(open(f))
    if d.get('finish_reason') == 'error' or d.get('parsed_json') is None: bad += 1
    else: ok += 1
print(f'extraction: {len(exts)} attempted | parsed {ok} | failed {bad}')

labs = [f for f in glob.glob('artifacts/gemma3_4b_pooled/labels/*.json') if not f.endswith('_summary.json')]
dom = defaultdict(lambda: [0,0]); tot = e = 0
for f in labs:
    d = json.load(open(f)); L = d.get('labels', [])
    n = len(L); ne = sum(int(x.get('is_error',0)) for x in L)
    dom[d.get('domain','?')][0] += n; dom[d.get('domain','?')][1] += ne
    tot += n; e += ne
print(f'labeled: {len(labs)} docs | {tot} fields | {e} errors ({e/max(tot,1):.1%})')
for dm,(n,ne) in sorted(dom.items()):
    print(f'  {dm:34} fields={n:5} errors={ne:5} ({ne/max(n,1):5.1%})')
print('')
print('Compare with Qwen-4B (~11% overall). Credit-agreement yield is the key')
print('number: the 12B lost 9 of 10 credit docs to OOM.')
"