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

set -e   # exit on any error
set -x   # print every command before running

source ~/NLP_Lab/setup_env.sh

which python
python --version
python -c "import sys; print(sys.executable)"
python -c "from probe_extraction.config import load_config; print('import OK')"

cd ~/NLP_Lab
nvidia-smi
python scripts/01_extract.py --config configs/exp_qwen35_4b_pymupdf.yaml 