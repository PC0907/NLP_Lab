#!/bin/bash
#SBATCH --partition=A100short
#SBATCH --time=1:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --job-name=install_a100
#SBATCH --output=logs/install-%j.out
#SBATCH --error=logs/install-%j.err

set -euo pipefail
module load Python/3.12.3
module load CUDA/12.4.0

# Fresh venv specifically for A100 (AMD EPYC) nodes
python -m venv ~/nlp_lab_a100
source ~/nlp_lab_a100/bin/activate

pip install --upgrade pip
# PyTorch — pip now detects the AMD CPU because this runs ON an A100 node
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
cd ~/NLP_Lab
pip install -r requirements.txt
pip install -e . --no-deps

echo "=== verify ==="
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"
python -c "from probe_extraction.config import load_config; print('package import OK')"

