#!/bin/bash
#SBATCH --partition=A40short
#SBATCH --time=1:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --job-name=insurance_4b
#SBATCH --output=logs/insurance_%j.out
#SBATCH --error=logs/insurance_%j.err
# Insurance-claims extraction (30 text docs, ~2.5k chars each). Small + text-native,
# so fast on A40. Full pipeline: extract -> label -> nested LODO.

source ~/NLP_Lab/setup_env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -euo pipefail
cd ~/NLP_Lab
echo "=== node/gpu ==="; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "=== [1/3] extract ==="
python scripts/01_extract.py --config configs/exp_qwen35_4b_insurance.yaml

echo "=== [2/3] label ==="
python scripts/02_label.py --config configs/exp_qwen35_4b_insurance.yaml

echo "=== error-rate check BEFORE committing to LODO ==="
python -c "
import json, glob
fs = [f for f in glob.glob('artifacts/qwen35_4b_insurance/labels/*.json') if not f.endswith('_summary.json')]
tot = err = 0
for f in fs:
    d = json.load(open(f)); L = d.get('labels', [])
    tot += len(L); err += sum(int(x.get('is_error',0)) for x in L)
print(f'docs={len(fs)} fields={tot} errors={err} rate={err/max(tot,1):.1%}')
print('-> if rate < ~8%, probe will be error-starved (like 9B-swimming); if 15-40%, good signal.')
"

echo "=== [3/3] nested LODO (all insurance, no domain exclusion needed) ==="
python scripts/05b_nested_lodo.py --config configs/exp_qwen35_4b_insurance.yaml --out-name nested_lodo.json

echo "=== done. results in artifacts/qwen35_4b_insurance/results/ ==="