#!/bin/bash
#SBATCH --partition=A40devel
#SBATCH --time=0:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --job-name=nlp_analysis
#SBATCH --output=logs/analysis-%j.out
#SBATCH --error=logs/analysis-%j.err

source ~/NLP_Lab/setup_env.sh
cd ~/NLP_Lab

python scripts/02_label.py --config configs/exp_qwen35_4b_docling.yaml
python scripts/03_train_probe.py --config configs/exp_qwen35_4b_docling.yaml
python scripts/04_evaluate.py --config configs/exp_qwen35_4b_docling.yaml

echo "=== RESULTS ==="
cat artifacts/qwen35_4b_docling/results/comparison.json 2>/dev/null || echo "no comparison.json"