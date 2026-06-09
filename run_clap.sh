#!/bin/bash
#SBATCH --partition=A40devel
#SBATCH --time=0:45:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --job-name=deepseek_clap
#SBATCH --output=logs/clap-%j.out
#SBATCH --error=logs/clap-%j.err

# ─────────────────────────────────────────────────────────────────────────────
# Stages 6–7: CLAP cross-layer attention probe — train + LODO CV.
# CPU-only. No GPU needed (the CLAP scripts run torch on CPU). A40devel = fast.
#
# What this does:
#   Stage 6 (06_train_clap.py): trains ONE CLAP transformer that attends over
#       all 14 captured layers jointly, with 5-fold CV + an 80/20 held-out
#       split. Writes probes/clap_probe.pt and results/clap_summary.json.
#   Stage 7 (07_lodo_clap.py): the honest, document-level metric. Trains CLAP
#       once per held-out document (LODO) and writes results/lodo_clap.json.
#       This is directly comparable to Stage 5's per-layer results/lodo_cv.json.
#
# Prerequisites (must already be on disk):
#   - run_deepseek_extract.sh  completed  -> activations/ + extractions/
#   - run_deepseek_analysis.sh completed  -> labels/ (Stage 2 produces these)
#
# Submit:
#   sbatch run_clap.sh
# ─────────────────────────────────────────────────────────────────────────────

source ~/NLP_Lab/setup_env.sh
cd ~/NLP_Lab

set -euo pipefail

CFG="configs/exp_deepseek_r1_7b_pooled.yaml"
EXP="deepseek_r1_7b_pooled"

echo "=== STAGE 06: Train CLAP cross-layer attention probe ==="
python scripts/06_train_clap.py --config "$CFG"

echo ""
echo "=== STAGE 07: LODO cross-validation for CLAP ==="
python scripts/07_lodo_clap.py --config "$CFG"

echo ""
echo "=== STAGES 06–07 COMPLETE ==="
echo ""
echo "=== CLAP SUMMARY (random-fold CV + 80/20 split) ==="
python -m json.tool "artifacts/$EXP/results/clap_summary.json" 2>/dev/null || \
    cat "artifacts/$EXP/results/clap_summary.json"

echo ""
echo "=== CLAP LODO (the honest, document-level metric) ==="
python -c "
import json
d = json.load(open('artifacts/$EXP/results/lodo_clap.json'))
m = d.get('lodo_auroc_mean'); s = d.get('lodo_auroc_std')
v = d.get('n_valid_folds'); n = d.get('n_docs')
if m is not None:
    print(f'CLAP LODO AUROC : {m:.4f} +/- {s:.4f}  ({v}/{n} valid folds)')
else:
    print('No valid LODO folds — check per_fold details in lodo_clap.json')
" 2>/dev/null || cat "artifacts/$EXP/results/lodo_clap.json"

echo ""
echo "=== COMPARE: per-layer logistic-regression LODO (Stage 5) ==="
echo "Best single-layer LODO AUROC from results/lodo_cv.json:"
python -c "
import json
d = json.load(open('artifacts/$EXP/results/lodo_cv.json'))
rows = [(int(k), v.get('lodo_auroc_mean')) for k, v in d.items()
        if isinstance(v, dict) and v.get('lodo_auroc_mean') is not None]
if rows:
    best = max(rows, key=lambda x: x[1])
    print(f'  best layer {best[0]}: {best[1]:.4f}')
    print('  (CLAP wins if its LODO AUROC >= this, WITHOUT manual layer selection)')
" 2>/dev/null || true