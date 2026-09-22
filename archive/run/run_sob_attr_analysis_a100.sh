#!/bin/bash
#SBATCH --partition=A100medium
#SBATCH --export=NONE
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --job-name=sob_attr_analysis
#SBATCH --output=logs/sob_attr_analysis-%j.out
#SBATCH --error=logs/sob_attr_analysis-%j.err

# CPU-only analysis for the attribution run: label (structure_aware), then the
# headline field-localized attribution LODO with paired significance. 16-way
# parallel folds, 4 layers. Prereq: run_sob_attr_extract_a100.sh finished.

module load Python/3.12.3
module load CUDA/12.4.0
source ~/nlp_lab_a100/bin/activate
export PYTHONPATH=$HOME/NLP_Lab/src:$PYTHONPATH
export HF_DATASETS_OFFLINE=1
export JOBLIB_TEMP_FOLDER=$TMPDIR

set -euo pipefail
cd ~/NLP_Lab

CFG="configs/exp_deepseek_r1_7b_sob_attr.yaml"

echo "=== STAGE 02: Label (structure_aware) ==="
python scripts/02_label.py --config "$CFG"

echo "=== STAGE 07: Field-localized reasoning attribution LODO ==="
python scripts/07_reasoning_attribution_lodo.py --config "$CFG" --layers 16 19 23 26 --jobs -1

echo ""
echo "=== ATTRIBUTION SUMMARY ==="
python -c "
import json
d=json.load(open('artifacts/deepseek_r1_7b_sob_attr/results/reasoning_attribution_lodo.json'))
print('value mentioned in trace: %.1f%% of fields' % (100*d['pct_value_mentioned']))
for m in ('per_doc_auroc_mean','pooled_oof_auroc'):
    print(m+':')
    for v in ('answer','fused_attr','fused_scalars','fused_both','scalars_only'):
        e=d['summary'][m][v]; a=e['best_auroc']
        print('  %-14s layer %s  AUROC %s' % (v, e['best_layer'], ('%.4f'%a) if a is not None else 'n/a'))
print('paired significance vs answer (per-doc):')
for v,s in d['significance_vs_answer'].items():
    p=s['p_value']; md=s['mean_delta']
    print('  %-14s delta=%+.4f  p=%s  n=%d' % (v, md if md is not None else float('nan'), ('%.4g'%p) if p is not None else 'n/a', s['n_pairs']))
"
echo "Done."
