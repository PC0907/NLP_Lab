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
module load Python/3.12.3
module load CUDA/12.4.0
source ~/nlp_lab_a100/bin/activate
export PYTHONPATH=$HOME/NLP_Lab/src:$PYTHONPATH
export HF_DATASETS_OFFLINE=1
set -euo pipefail
cd ~/NLP_Lab
CFG="configs/exp_deepseek_r1_7b_sob.yaml"; EXP="deepseek_r1_7b_sob"
python scripts/02_label.py           --config "$CFG"
python scripts/03_train_probe.py     --config "$CFG"
python scripts/04_evaluate.py        --config "$CFG"
python scripts/05_lodo_cv.py         --config "$CFG"
python scripts/06_reasoning_fusion_lodo.py --config "$CFG"
echo "=== ERROR-DEFINITION CHECK ==="
python -m json.tool "artifacts/$EXP/labels/_definition_comparison.json" 2>/dev/null || true
echo "Done. tar artifacts/$EXP/ and send it back."
