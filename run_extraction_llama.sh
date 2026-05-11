#!/bin/bash
#SBATCH --partition=A100short
#SBATCH --time=8:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --job-name=llama31_8b
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

source ~/NLP_Lab/setup_env.sh

cd ~/NLP_Lab
nvidia-smi

python scripts/01_extract.py --config configs/exp_llama31_8b.yaml
python scripts/02_label.py --config configs/exp_llama31_8b.yaml
python scripts/03_train_probe.py --config configs/exp_llama31_8b.yaml
python scripts/04_evaluate.py --config configs/exp_llama31_8b.yaml