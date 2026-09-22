#!/bin/bash
#SBATCH --partition=A100medium
#SBATCH --export=NONE
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --job-name=sob_stage06
#SBATCH --output=logs/sob_stage06-%j.out
#SBATCH --error=logs/sob_stage06-%j.err

# Headline reasoning-fusion LODO only. Stages 02-04 already done; Stage 05 is
# redundant (Stage 06's 'answer' variant IS answer-only LODO). CPU-only, 16-way
# parallel folds, 4 best layers. ~30 min instead of ~20 h.
module load Python/3.12.3
module load CUDA/12.4.0
source ~/nlp_lab_a100/bin/activate
export PYTHONPATH=$HOME/NLP_Lab/src:$PYTHONPATH
export HF_DATASETS_OFFLINE=1
export JOBLIB_TEMP_FOLDER=$TMPDIR
set -euo pipefail
cd ~/NLP_Lab

CFG="configs/exp_deepseek_r1_7b_sob.yaml"
python scripts/06_reasoning_fusion_lodo.py --config "$CFG" --layers 16 19 23 26 --jobs -1

echo "=== SUMMARY ==="
python -c "
import json
d=json.load(open('artifacts/deepseek_r1_7b_sob/results/reasoning_fusion_lodo.json'))
s=d['summary']
for m in ('per_doc_auroc_mean','pooled_oof_auroc'):
    print(m+':')
    for v in ('answer','fused_mean','fused_last','fused_both'):
        e=s[m][v]; a=e['best_auroc']
        print(f'  {v:<12} layer {e[\"best_layer\"]}  AUROC ' + (f'{a:.4f}' if a is not None else 'n/a'))
"
