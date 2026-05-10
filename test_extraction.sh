#!/bin/bash
#SBATCH --partition=A40devel
#SBATCH --time=0:30:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --job-name=nlp_test
#SBATCH --output=logs/test-%j.out
#SBATCH --error=logs/test-%j.err

source ~/NLP_Lab/setup_env.sh

cd ~/NLP_Lab
nvidia-smi
python scripts/01_extract.py --config configs/exp_qwen35_4b_pymupdf.yaml --limit 1