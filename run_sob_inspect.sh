#!/bin/bash
#SBATCH --partition=A40short
#SBATCH --time=0:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --job-name=sob_inspect
#SBATCH --output=logs/sob_inspect_%j.out
#SBATCH --error=logs/sob_inspect_%j.err
# CPU-only: inspect SOB's real columns + verify the loader. No GPU requested.
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
source ~/NLP_Lab/setup_env.sh
set -euo pipefail
cd ~/NLP_Lab

echo "=== SOB raw columns (text config) ==="
python -c "
from datasets import load_dataset
ds = load_dataset('interfaze-ai/sob', 'text')
split = list(ds.keys())[0]; d = ds[split]
print('splits:', list(ds.keys()), '| n:', len(d))
print('columns:', d.column_names)
r = d[0]
for k, v in r.items():
    s = str(v); print(f'  {k}: len={len(s)} | {s[:150]}')
"

echo ""
echo "=== loader verification (3 docs) ==="
python -c "
from probe_extraction.data.sob import SOBench
b = SOBench(max_documents=3)
print('docs:', len(b))
for d in b:
    props = list(d.schema.get('properties',{}).keys())[:5] if isinstance(d.schema, dict) else 'n/a'
    print(' ', d.doc_id, '| text=', len(d.text), '| schema_keys=', props, '| gold_keys=', list(d.gold.keys())[:5])
    print('    text preview:', d.text[:150].replace(chr(10),' '))
"