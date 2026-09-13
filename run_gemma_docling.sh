#!/bin/bash
#SBATCH --partition=A100short
#SBATCH --time=8:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --export=NONE
#SBATCH --job-name=gem_pool_dl
#SBATCH --output=logs/gem_pool_dl_%j.out
#SBATCH --error=logs/gem_pool_dl_%j.err
#
# Gemma-3-4B across four ExtractBench domains, parsed with Docling.
#
# Follows the swimming smoke test, where Docling cut the error rate from ~58%
# to 34.4% (table2 alone: 86% -> 17%). Still above Gemma-3-12B's 0-13% on the
# same documents, so the parser was part of the problem and capacity is the
# rest -- but 34% is a trainable rate, which it was not before.
#
# Comparison target: artifacts/qwen35_4b_pooled_docling (26 docs, same parser).
# Same parser on both sides means no parser confound; the intersection of
# documents both models completed is the fair comparison set.
#
# Stops after labeling. Check document yield and error rates before running
# probes -- no point training on labels that turn out to be wrong.

unset SLURM_EXPORT_ENV
source ~/NLP_Lab/setup_env_a100.sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

set -euo pipefail
cd ~/NLP_Lab
echo "=== gpu ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "=== [1/2] extract (4 domains, docling) ==="
python scripts/01_extract.py --config configs/exp_gemma3_4b_pooled_docling.yaml

echo "=== [2/2] label ==="
python scripts/02_label.py --config configs/exp_gemma3_4b_pooled_docling.yaml

echo ""
echo "=== YIELD AND ERROR RATES, vs Qwen on the same parser ==="
python -c "
import json, glob
from collections import defaultdict

def summarise(d):
    dom = defaultdict(lambda: [0, 0])
    docs = set()
    for f in glob.glob(f'artifacts/{d}/labels/*.json'):
        if f.endswith('_summary.json'):
            continue
        x = json.load(open(f))
        L = x.get('labels', [])
        k = x.get('domain', '?')
        dom[k][0] += len(L)
        dom[k][1] += sum(int(i.get('is_error', 0)) for i in L)
        docs.add(f.split('/')[-1].replace('.json', ''))
    return dom, docs

g, gdocs = summarise('gemma3_4b_pooled_docling')
q, qdocs = summarise('qwen35_4b_pooled_docling')

print(f'{\"domain\":34} {\"Gemma-3-4B\":>22}   {\"Qwen3.5-4B\":>22}')
for k in sorted(set(g) | set(q)):
    gn, ge = g.get(k, [0, 0])
    qn, qe = q.get(k, [0, 0])
    gs = f'{gn:5} fields {ge/max(gn,1):6.1%}' if gn else '                  ---'
    qs = f'{qn:5} fields {qe/max(qn,1):6.1%}' if qn else '                  ---'
    print(f'{k:34} {gs:>22}   {qs:>22}')

gt = sum(v[0] for v in g.values()); getot = sum(v[1] for v in g.values())
qt = sum(v[0] for v in q.values()); qetot = sum(v[1] for v in q.values())
print()
print(f'Gemma: {len(gdocs):3} docs, {gt:5} fields, {getot/max(gt,1):.1%} errors')
print(f'Qwen : {len(qdocs):3} docs, {qt:5} fields, {qetot/max(qt,1):.1%} errors')
print()
both = sorted(gdocs & qdocs)
print(f'Documents completed by BOTH models: {len(both)}')
print('This intersection is the fair comparison set -- same documents, same')
print('parser, different model families. Write it out for --include-docs-file:')
with open('intersection_gemma_qwen_docling.txt', 'w') as fh:
    for d in both:
        fh.write(d + '\n')
print('  wrote intersection_gemma_qwen_docling.txt')
"