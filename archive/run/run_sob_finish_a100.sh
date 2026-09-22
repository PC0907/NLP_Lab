#!/bin/bash
#SBATCH --partition=A100medium
#SBATCH --export=NONE
#SBATCH --time=5:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --job-name=sob_finish_a100
#SBATCH --output=logs/sob_finish_a100-%j.out
#SBATCH --error=logs/sob_finish_a100-%j.err

# Resume: Stages 02-04 already completed & saved. Only run the two slow LODO
# stages that timed out. A100medium (not devel) + 5h wall time.
module load Python/3.12.3
module load CUDA/12.4.0
source ~/nlp_lab_a100/bin/activate
export PYTHONPATH=$HOME/NLP_Lab/src:$PYTHONPATH
export HF_DATASETS_OFFLINE=1
set -euo pipefail
cd ~/NLP_Lab

CFG="configs/exp_deepseek_r1_7b_sob.yaml"
EXP="deepseek_r1_7b_sob"

echo "=== STAGE 05: Per-layer LODO ==="
python scripts/05_lodo_cv.py --config "$CFG"
echo "=== STAGE 06: Reasoning-fusion LODO (HEADLINE) ==="
python scripts/06_reasoning_fusion_lodo.py --config "$CFG"

echo ""
echo "=== REASONING-FUSION SUMMARY ==="
python -c "
import json
d = json.load(open('artifacts/$EXP/results/reasoning_fusion_lodo.json'))
s = d['summary']
for metric in ('per_doc_auroc_mean','pooled_oof_auroc'):
    print(metric+':')
    for v in ('answer','fused_mean','fused_last','fused_both'):
        e = s[metric][v]; a = e['best_auroc']
        print(f'  {v:<12} layer {e[\"best_layer\"]}  AUROC ' + (f'{a:.4f}' if a is not None else 'n/a'))
" 2>/dev/null || cat "artifacts/$EXP/results/reasoning_fusion_lodo.json"
echo "Done."
