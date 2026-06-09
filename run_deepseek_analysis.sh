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
echo "=== STAGES 02–05 COMPLETE ==="
echo ""
echo "=== PER-LAYER LODO RESULTS ==="
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
echo "Next step: sbatch run_clap.sh"