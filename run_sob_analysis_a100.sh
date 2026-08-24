#!/bin/bash
#SBATCH --partition=A100devel
#SBATCH --export=NONE
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --job-name=sob_analysis_a100
#SBATCH --output=logs/sob_analysis_a100-%j.out
#SBATCH --error=logs/sob_analysis_a100-%j.err

# Stages 02-06 for the SOB run. CPU-only work, but uses the AMD a100 venv so it
# runs on an A100 node (keeps one venv for the whole A100 workflow).
# Prereq: run_sob_extract_a100.sh finished.

module load Python/3.12.3
module load CUDA/12.4.0
source ~/nlp_lab_a100/bin/activate
export PYTHONPATH=$HOME/NLP_Lab/src:$PYTHONPATH
export HF_DATASETS_OFFLINE=1

set -euo pipefail
cd ~/NLP_Lab

CFG="configs/exp_deepseek_r1_7b_sob.yaml"
EXP="deepseek_r1_7b_sob"

echo "=== STAGE 02: Label (structure_aware) ==="
python scripts/02_label.py --config "$CFG"
echo "=== STAGE 03: Train per-layer probes ==="
python scripts/03_train_probe.py --config "$CFG"
echo "=== STAGE 04: Evaluate vs baselines ==="
python scripts/04_evaluate.py --config "$CFG"
echo "=== STAGE 05: Per-layer LODO ==="
python scripts/05_lodo_cv.py --config "$CFG"
echo "=== STAGE 06: Reasoning-fusion LODO ==="
python scripts/06_reasoning_fusion_lodo.py --config "$CFG"

echo ""
echo "=== ERROR-DEFINITION CHECK (strict vs auto vs structure_aware) ==="
python -m json.tool "artifacts/$EXP/labels/_definition_comparison.json" 2>/dev/null || true
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
echo ""
echo "Done. tar artifacts/$EXP/ and send it back."
