#!/bin/bash
#SBATCH --partition=A40short
#SBATCH --time=4:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --job-name=qwen35_pymupdf
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

source ~/NLP_Lab/setup_env.sh

cd ~/NLP_Lab
nvidia-smi

python scripts/01_extract.py --config configs/exp_qwen35_4b_swimming.yaml
python scripts/02_label.py --config configs/exp_qwen35_4b_swimming.yaml
python scripts/03_train_probe.py --config configs/exp_qwen35_4b_swimming.yaml
python scripts/04_evaluate.py --config configs/exp_qwen35_4b_swimming.yaml