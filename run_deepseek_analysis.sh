#!/bin/bash
#SBATCH --partition=A40devel
#SBATCH --time=0:45:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --job-name=deepseek_analysis
#SBATCH --output=logs/deepseek_analysis-%j.out
#SBATCH --error=logs/deepseek_analysis-%j.err

# ─────────────────────────────────────────────────────────────────────────────
# Stages 2–5: Label → Train probe → Evaluate → LODO CV
# CPU-only. No GPU needed. Runs on A40devel (fast queue).
#
# Prerequisite: run_deepseek_extract.sh must have completed successfully.
#
# Submit:
#   sbatch run_deepseek_analysis.sh
#
# After this completes, run:
#   sbatch run_clap.sh
# ─────────────────────────────────────────────────────────────────────────────

source ~/NLP_Lab/setup_env.sh
cd ~/NLP_Lab

set -euo pipefail

CFG="configs/exp_deepseek_r1_7b_pooled.yaml"
EXP="deepseek_r1_7b_pooled"

echo "=== STAGE 02: Label extracted fields ==="
python scripts/02_label.py --config "$CFG"

echo ""
echo "=== STAGE 03: Train per-layer logistic regression probes ==="
python scripts/03_train_probe.py --config "$CFG"

echo ""
echo "=== STAGE 04: Evaluate probes vs baselines ==="
python scripts/04_evaluate.py --config "$CFG"

echo ""
echo "=== STAGE 05: LODO cross-validation (per-layer logistic regression) ==="
python scripts/05_lodo_cv.py --config "$CFG"

echo ""
echo "=== STAGE 06: Reasoning-fusion LODO (answer vs reasoning-fused) ==="
python scripts/06_reasoning_fusion_lodo.py --config "$CFG"

echo ""
echo "=== STAGES 02–06 COMPLETE ==="
echo ""
echo "=== ERROR-DEFINITION CHECK (strict vs auto vs structure_aware) ==="
echo "    The whole point: structure_aware should be FAR below strict's ~95%."
python -m json.tool "artifacts/$EXP/labels/_definition_comparison.json" 2>/dev/null || \
    cat "artifacts/$EXP/labels/_definition_comparison.json" 2>/dev/null || true

echo ""
echo "=== PER-LAYER LODO RESULTS (answer-token baseline) ==="
python -c "
import json
with open('artifacts/$EXP/results/lodo_cv.json') as f:
    d = json.load(f)
print(f'{'Layer':<8} {'LODO AUROC':<14} {'Std':<10} {'Valid folds'}')
print('-' * 45)
for layer, res in sorted(d.items(), key=lambda x: int(x[0])):
    m = res.get('lodo_auroc_mean')
    s = res.get('lodo_auroc_std', 0.0)
    v = res.get('n_valid_folds', 0)
    n = res.get('n_docs', '?')
    if m is not None:
        print(f'{layer:<8} {m:.4f}         {s:.4f}     {v}/{n}')
    else:
        print(f'{layer:<8} n/a')
" 2>/dev/null || cat "artifacts/$EXP/results/lodo_cv.json"

echo ""
echo "=== REASONING-FUSION SUMMARY (answer vs fused; per_doc vs pooled_oof) ==="
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
echo "Done. Hand artifacts/$EXP/ back for analysis."