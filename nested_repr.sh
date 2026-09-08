#!/bin/bash
#SBATCH --partition=A40short
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --job-name=nested_repro
#SBATCH --output=logs/nested_repro_%j.out
#SBATCH --error=logs/nested_repro_%j.err
# CPU-only: re-run nested LODO to emit pooled_oof_auroc. No GPU.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
source ~/NLP_Lab/setup_env.sh
set -euo pipefail
cd ~/NLP_Lab

echo "########## ALLTOKENS ##########"
python scripts/05b_nested_lodo.py \
    --config configs/exp_qwen35_4b_pooled_alltokens.yaml \
    --layers 14 16 18 20 22 \
    --exclude-domains finance/10kq \
    --out-name nested_lodo_repro.json

echo ""
echo "########## POOLED ##########"
python scripts/05b_nested_lodo.py \
    --config configs/exp_qwen35_4b_pooled.yaml \
    --layers 1 4 8 12 14 16 18 20 22 24 26 28 30 32 \
    --exclude-domains finance/10kq \
    --out-name nested_lodo_repro.json

echo ""
echo "########## SUMMARY ##########"
python -c "
import json
for tree in ['qwen35_4b_pooled_alltokens','qwen35_4b_pooled']:
    d=json.load(open(f'artifacts/{tree}/results/nested_lodo_repro.json'))
    print(tree)
    print('  per-fold  ', round(d['auroc_mean'],4), '+/-', round(d['auroc_std'],4))
    print('  pooled-OOF', round(d['pooled_oof_auroc'],4))
    print('  AUPRC     ', round(d['auprc'],4))
    print('  folds     ', d['n_folds'], ' layers', d['layers_selected'])
"